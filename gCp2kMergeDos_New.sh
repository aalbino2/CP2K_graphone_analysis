#!/bin/bash
# 9 Nov, 2009
# Reads all the .pdos files and creates new TDOS files named $InputName.pdos

while ( [ $# != 0 ] ); do
 case $1 in
 -help| -h| --help)
  echo '==============================================================='
  echo 'Merge the .pdos files'
  echo ''
  echo 'Usage: gCp2kMergeDos [ FileNames ]'
  echo '==============================================================='
  exit 0
 ;;
 *)
  FileNames="$FileNames $1"
 ;;
 esac
 shift
done

if [[ ! $FileNames ]]; then
 FileNames=$( ls *k*.pdos 2> /dev/null )
fi
 echo $FileNames

# Check and merge the unrestricted files
echo "-----"
echo "Unrestricted pdos script... "
for FileName in $FileNames; do
 Tmp=$( echo $FileName | grep ALPHA ) # ]; then
  echo "* found file in folder:"
  echo $FileName, $Tmp
 if [ $Tmp ]; then
  RootName=$( echo $FileName | sed 's/ALPHA_//' | sed 's/BETA_//' | sed 's/-1\.pdos//' )
  # echo $FileName $RootName
  RootNames="$RootNames\n$RootName"
 else
  Tmp=$( echo $FileName | grep BETA )
  if [ ! $Tmp ]; then
   RestrictedFiles="$RestrictedFiles\n$FileName"
   # echo $FileName, $RestrictedFiles
  fi
 fi
done
# exit 0
UnrestrictedFiles=$( echo -e $RootNames| sort -u )
echo "Unrestricted files found:  $UnrestrictedFiles"
# exit 0
echo "-----"

echo "Restricted pdos script... "
for UnrestrictedName in $UnrestrictedFiles; do
 if [ -f $UnrestrictedName-1.pdos ]; then
  echo "$UnrestrictedName-1.pdos already exists. Skipping"
 else
  Alpha=$( echo $UnrestrictedName | sed 's/-\(k*\)/-ALPHA_\1/' | sed 's/$/-1.pdos/' )
  Beta=$( echo $UnrestrictedName | sed 's/-\(k*\)/-BETA_\1/' | sed 's/$/-1.pdos/' )
  a=$( head -1 $Alpha | awk '{print $(NF-1)}' )
  b=$( head -1 $Beta | awk '{print $(NF-1)}' )
  Larger=$( echo $a $b | awk '{if( $1 >= $2 ){print $1}else{print $2}}' )

############# AGGIUNTO
   if (( $(echo "${Larger} < 0" | bc -l) )); then
   echo "*"
   echo "E fermi: alpha="  $a "beta=" $b
   echo "*"
   head -2 $Alpha | sed "s/-[0-9.]* a.u./$Larger a.u./" > $UnrestrictedName-1.pdos
   else
   echo "*"
   echo "E fermi: alpha="  $a "beta=" $b
   echo "*"
   head -2 $Alpha | sed "s/[0-9.]* a.u./$Larger a.u./" > $UnrestrictedName-1.pdos
   fi
###########################

  grep -v \# $Alpha > _tmp.pdos
  grep -v \# $Beta >> _tmp.pdos
  sort -k 2 -n _tmp.pdos >> $UnrestrictedName-1.pdos
  rm _tmp.pdos
 fi
 RestrictedFiles="$RestrictedFiles\n$UnrestrictedName-1.pdos"
done

RestrictedFiles=$( echo -e $RestrictedFiles | sort -u )
 echo "Restricted files found: $RestrictedFiles"

for FileName in $RestrictedFiles; do
  RootName=$( echo $FileName | sed 's/-k[0-9]*-1.pdos//' )
  # echo "--- $FileName --- $RootName"
  RestrictedNames="$RestrictedNames\n$RootName"
done
RestrictedNames="${RestrictedNames:2}" # removes leading "\n"
RestrictedNamesUnique=$( echo -e $RestrictedNames | sort -u )
echo "Name roots identified: $RestrictedNamesUnique"
for File in $RestrictedNamesUnique; do
 echo "Searching *k*-1.pdos files with root name: $File"
 NumOfFiles=$( ls ${File}-k*.pdos 2>/dev/null | wc -l )
 NumOfFields=$( grep -v \# ${File}-k1-1.pdos | head -1 | awk '{print NF-3}' )
 echo "no. of files: $NumOfFiles; no. of fields in each file: $NumOfFields"
 grep -v \# ${File}-k1-1.pdos | awk '{printf "%8d\t%10.6f\t%10.6f\n", $1, $2, $3}' > _Tmp0

 FileNames=$( ls ${File}-k*.pdos 2>/dev/null )
 for i in $FileNames; do
  grep -v \# $i | awk '{
  for (j=4; j<NF; j++){
   printf $j"\t"
  }
   print $NF
  }' > _tmp$i
 done
 head -2 ${File}-k1-1.pdos > ${File}-1.pdos
 paste _tmp* | awk -v NumOfFiles=$NumOfFiles -v NumOfFields=$NumOfFields \
 '{
   for (j=0; j<NumOfFiles; j++){
    for (i=1; i<=NumOfFields; i++){
     el[i]=el[i]+$(i+(j*NumOfFields))
     # print i, el[i]
     if (j==(NumOfFiles-1)){
      printf "%12.8f\t  ", el[i]
      el[i]=0
     }
    }
   }
   printf "\n"
 }' > _TmpAll
 paste _Tmp0 _TmpAll >> ${File}-1.pdos
 rm _Tmp0 _TmpAll _tmp*
done
echo "-----"
echo "Done"
