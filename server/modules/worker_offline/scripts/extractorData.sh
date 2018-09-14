#!/bin/bash

#Need lattice-to-nbest file in param
dataFile=$1

#Init value
declare -a utteranceValue
declare -a acousticScore
declare -a languageScore

countUtterance=1
previousId=1
acousticScore[previousId]=0
languageScore[previousId]=0


#Position of important colonne 
posName=$2
posAcoustic=2
posLanguage=3
posUtterance=4



while read -r line
do
	name="$line"
	arrLine=($line)
	currentId="${arrLine[$posName]##*-}"

	if [ "$currentId" -eq "$previousId" ]; then
		#Manage all data value before storage
		countUtterance=$((countUtterance + 1))
		utteranceValue[$currentId]="${utteranceValue[currentId]} ${arrLine[$posUtterance]}"
		acousticScore[$currentId]=`echo ${acousticScore[currentId]} + ${arrLine[$posAcoustic]} | bc`
		languageScore[$currentId]=`echo ${languageScore[currentId]} + ${arrLine[$posLanguage]} | bc`	
	else
		#Do stuff on the previous segment before swap
		acousticScore[$previousId]="$(echo "scale=2; ${acousticScore[$previousId]}/$countUtterance" | bc | sed -e 's/^0*//' -e 's/^\./0./')"
		languageScore[$previousId]="$(echo "scale=2; ${languageScore[$previousId]}/$countUtterance" | bc | sed -e 's/^0*//' -e 's/^\./0./')"

		#Init for the next seglment
		previousId=$currentId
		countUtterance=0
		concatUterance=""
		acousticScore[previousId]=0
		languageScore[previousId]=0

		#Init data for the uterance
		countUtterance=$((countUtterance + 1))
		utteranceValue[$currentId]="${utteranceValue[currentId]}${arrLine[$posUtterance]}"
		acousticScore[$currentId]=`echo ${acousticScore[currentId]} + ${arrLine[$posAcoustic]} | bc`
		languageScore[$currentId]=`echo ${languageScore[currentId]} + ${arrLine[$posLanguage]} | bc`	

	fi
done < "$dataFile"

#Need to manage the last data
acousticScore[$previousId]="$(echo "scale=2; ${acousticScore[$previousId]}/$countUtterance" | bc | sed -e 's/^0*//' -e 's/^\./0./')"
languageScore[$previousId]="$(echo "scale=2; ${languageScore[$previousId]}/$countUtterance" | bc | sed -e 's/^0*//' -e 's/^\./0./')"


#start to 1, no id 0 stored

if [ "$posName" -eq 1 ]; then
	echo -n "{\"utterance\":\"${utteranceValue[1]}\",\"acousticScore\":${acousticScore[1]},\"languageScore\":${languageScore[1]}}"
else 
	echo -n "{\"hypotheses\":["
	for i in `seq 1 $previousId`; do
			if [ -z "${acousticScore[$i]}" ]; then
					acousticScore[$i]=0.0
			fi

			if [ -z "${languageScore[$i]}" ]; then
					languageScore[$i]=0.0
			fi

			if [ "$i" -eq "$previousId" ]; then
				echo -n "{\"utterance\":\"${utteranceValue[$i]}\",\"acousticScore\":${acousticScore[$i]},\"languageScore\":${languageScore[$i]}}"
			else
				echo -n "{\"utterance\":\"${utteranceValue[$i]}\",\"acousticScore\":${acousticScore[$i]},\"languageScore\":${languageScore[$i]}},"
			fi
	done
	echo -n "]}"
fi