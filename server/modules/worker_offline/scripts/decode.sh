#!/bin/bash

# Copyright 2018 Linagora (author: Ilyes Rebai)
# LinSTT project


lvcsrRootDir=/opt/speech-to-text
export KALDI_ROOT=$lvcsrRootDir/tools/kaldi
export PATH=$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/kwsbin:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin/:$KALDI_ROOT/src/lmbin/:$PWD:$PATH

# Begin configuration section.
mfcc_config=conf/mfcc.conf
vad_config=conf/vad.conf
ivector=false
type=
compress=true
cmd=run.pl
# End configuration section.

. parse_options.sh || exit 1;

if [ $# != 10 ]; then
   echo "Usage: $0 [options] <model-dir> <wav-file> <do-ctm> <max_active> <beam> <lattice_beam> <acwt> <decoder> <nj> <num_threads>";
   echo "e.g.: $0 model-soft test.wav True"
   echo "Options: "
   echo "  --mfcc-config <config-file>                      # config passed to compute-mfcc-feats "
   echo "  --vad-config <config-file>                       # config passed to compute-vad-energy"
   echo "  --nj <nj>                                        # number of parallel jobs for feature extraction"
   echo "  --decode-nj <nj>                                 # number of parallel jobs for decoding"
   echo "  --num-threads <n>                                # number of threads to use, default 1."
   echo "  --write-utt2num-frames <true|false>     # If true, write utt2num_frames file."
   echo "  --acwt <float>                                   # acoustic scale used for lattice generation "
   exit 1;
fi
amdir=$1
curentFile=$2
indiceData=$3
max_active=$4
beam=$5
lattice_beam=$6
acwt=$7
decoder=$8
nj=$9
num_threads=${10}

wavdir=$lvcsrRootDir/wavs
sysdir=$(readlink -f $1)
sysRootName=$(echo $(basename $sysdir)|cut -f1 -d"=")
file="$wavdir/$curentFile"
fileRootName=$(basename $file .wav)
datadir=$lvcsrRootDir/kaldi_input_data/$fileRootName
transdir=$amdir/decode_$fileRootName
mfcc_config=$amdir"/"$mfcc_config
vad_config=$amdir"/"$vad_config

function confidence_score {
  confidence=$lvcsrRootDir/trans/confidence_score.log
  decode=$1
  file=$2
  $cmd JOB=1:1 $confidence \
    lattice-to-ctm-conf2 --acoustic-scale=$acwt  \
     ark:$lvcsrRootDir/trans/$file.lat \
     $lvcsrRootDir/trans/log/sentence_confidence_score.txt
  #Bayes_Risk
  br=$(cat $confidence | grep "For utterance $file," | cut -d ',' -f2 | cut -d ' ' -f4)
  #AVG_Confidence_per_Word
  cw=$(cat $confidence | grep "For utterance $file," | cut -d ',' -f3 | cut -d ' ' -f5)
  #STD_per_sentence
  std=$(cat $confidence | grep "For utterance $file," | cut -d ',' -f4 | cut -d ' ' -f5)
  #Transcription
  text=$(cat $decode | sed -e "s: *$::g" -e "s:^ *::g")
  echo -n "{\"utterance\":\"$text\",\"br\":$br,\"cw\":$cw,\"std\":$std,\"message\":\"\"}" > $decode
}

[ -d $datadir ] || mkdir -p $datadir
mkdir -p $lvcsrRootDir/trans


if [ $decoder == "dnn" ]; then

decode_conf=$amdir/online/conf/online_nnet2_decoding.conf
decode_mdl=$amdir/online/final.mdl
decode_graph=$amdir/online/graph/HCLG.fst
decode_words=$amdir/online/graph/words.txt

if $ivector; then
  if [ -d $amdir/online/conf_ivector ]; then
    decode_conf=$amdir/online/conf_ivector/online_nnet2_decoding.conf
    decode_mdl=$amdir/online/final.ivector.mdl
  else
    error="You have activated ivector option. However, the decoding model is not designed to use this option (check the decode.cfg comments)."
    echo -n "{\"utterance\":\"ERROR!!!\",\"message\":\"$error\"}" > $lvcsrRootDir/trans/decode_$fileRootName.log
    exit 1
  fi
fi

if [ $type != "none" ]; then
  if [ ! -f $amdir/online/graph_${type}/HCLG.fst ]; then
    error="You used $type option. However, the decoding model is not designed to use this option (check the decode.cfg comments)."
    echo -n "{\"utterance\":\"ERROR!!!\",\"message\":\"$error\"}" > $lvcsrRootDir/trans/decode_$fileRootName.log
    exit 1
  else
    decode_graph=$amdir/online/graph_${type}/HCLG.fst
    decode_words=$amdir/online/graph_${type}/words.txt
  fi
fi

mkdir -p $lvcsrRootDir/trans/log
$cmd JOB=1:1 $lvcsrRootDir/trans/log/decode_$fileRootName.log \
  online2-wav-nnet2-latgen-faster --do-endpointing=false \
     --online=false \
     --config=$decode_conf \
     --max-active=$max_active --beam=$beam --lattice-beam=$lattice_beam \
     --acoustic-scale=$acwt \
     --word-symbol-table=$decode_words \
      $decode_mdl \
      $decode_graph \
      "ark:echo $fileRootName $fileRootName|" "scp:echo $fileRootName $file|" \
      ark:$lvcsrRootDir/trans/$fileRootName.lat

  error=$(cat $lvcsrRootDir/trans/log/decode_$fileRootName.log | grep "^ERROR " | head -n 1)
  if [[ $error == "" ]]; then
    cat $lvcsrRootDir/trans/log/decode_$fileRootName.log | grep "^$fileRootName" | cut -d ' ' -f2- > $lvcsrRootDir/trans/decode_$fileRootName.log
    if [ "$indiceData" = "True" ] ; then
       confidence_score $lvcsrRootDir/trans/decode_$fileRootName.log $fileRootName
    else
       text=$(cat $lvcsrRootDir/trans/decode_$fileRootName.log | sed -e "s: *$::g" -e "s:^ *::g")
       echo -n "{\"utterance\":\"$text\",\"message\":\"\"}" > $lvcsrRootDir/trans/decode_$fileRootName.log
    fi
  else
    echo -n "{\"utterance\":\"ERROR!!!\",\"message\":\"$error\"}" > $lvcsrRootDir/trans/decode_$fileRootName.log
  fi

else

#        echo "doing Speaker diaritization : segment extraction"
#        java -Xmx2024m -jar $lvcsrRootDir/tools/lium_spkdiarization-8.4.1.jar  \
#            --fInputMask=$file --sOutputMask=$datadir/$fileRootName.seg --doCEClustering $fileRootName
	duration=`soxi -D $file`
	echo "$fileRootName 1 0 $duration M S U S0" | sort -nk3 > $datadir/$fileRootName.seg


	############################################## Generate kaldi input for offline decoding ##############################################
	# file gen: segments, utt2spk, spk2utt, wav.scp

	# Gen segments file
	awk '$1 !~ /^;;/ {print $1"-"$8"-"$3"-"($3+$4)" "$1" "$3" "($3+$4)}' $datadir/$fileRootName.seg | sort -nk3 > $datadir/segments
	# Gen utt2spk file
	awk '{split($1,a,"-"); print $1" "a[2]  }'  $datadir/segments > $datadir/utt2spk
	# Gen spk2utt file
	cat $datadir/utt2spk | utt2spk_to_spk2utt.pl > $datadir/spk2utt
	# Gen wav.scp
	(for tag in $(cut -f1 -d"-" $datadir/spk2utt | cut -f2 -d" "); do
	    echo "$tag sox $file -t wav -r 16000 -c 1 - |"
	done) > $datadir/wav.scp
	cat $datadir/wav.scp | awk '{ print $1, $1, "A"; }' > $datadir/reco2file_and_channel
	echo validate_data_dir.sh
	validate_data_dir.sh --no-text --no-feats $datadir
	fix_data_dir.sh $datadir


	############################################## Generate Feature for each segments ##############################################
	features=$datadir/features
	logdir=$datadir/log
	name=$fileRootName
	data=$datadir
	mkdir -p $features $logdir
	# Compute MFCC #
	split_segments=""
	for n in $(seq $nj); do
	  split_segments="$split_segments $logdir/segments.$n"
	done
	split_scp.pl $data/segments $split_segments || exit 1;
	$cmd JOB=1:$nj $logdir/make_mfcc_${name}.JOB.log \
	    extract-segments scp,p:$data/wav.scp $logdir/segments.JOB ark:- \| \
	    compute-mfcc-feats $vtln_opts --verbose=2 --config=$mfcc_config ark:- ark:- \| \
	    copy-feats --compress=$compress ark:- \
	      ark,scp:$features/raw_mfcc_$name.JOB.ark,$features/raw_mfcc_$name.JOB.scp \
	    || exit 1;
	cat $features/raw_mfcc_$name.*.scp > $data/feats.scp || exit 1

	# Compute VAD #
	utils/split_data.sh $data $nj || exit 1;
	sdata=$data/split$nj;
	$cmd JOB=1:$nj $logdir/vad_${name}.JOB.log \
	    compute-vad --config=$vad_config scp:$sdata/JOB/feats.scp ark,t,scp:$features/vad_${name}.JOB.txt,$features/vad_${name}.JOB.scp \
	    || exit 1;
	cat $features/vad_${name}.*.scp > $data/vad.scp

	# Compute CMVN #
	! select-voiced-frames scp:$data/feats.scp scp,s,cs:$data/vad.scp ark:- | compute-cmvn-stats --spk2utt=ark:$data/spk2utt ark:- ark,scp:$features/cmvn_$name.ark,$features/cmvn_$name.scp \
    	    2> $logdir/cmvn_$name.log && echo "Error computing CMVN stats" && exit 1;
	cp $features/cmvn_$name.scp $data/cmvn.scp || exit 1;

	rm -r $sdata $logdir


	if [ ! -f $transdir/trans.1 ]; then
	    srcdir=$amdir
	    [ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"
	    utils/split_data.sh $data $nj || exit 1;
	    sdata=$data/split$nj;
	    splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
	    cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`
	    feats="ark,s,cs:select-voiced-frames scp:$sdata/JOB/feats.scp scp,s,cs:$sdata/JOB/vad.scp ark:- | apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp ark:- ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
	if [ -f "$graphdir/num_pdfs" ]; then
	    [ "`cat $graphdir/num_pdfs`" -eq `am-info --print-args=false $model | grep pdfs | awk '{print $NF}'` ] || \
	    { echo "Mismatch in number of pdfs with $model"; exit 1; }
	fi
	$cmd --num-threads $num_threads JOB=1:$nj $transdir/log/decode.JOB.log \
	    gmm-latgen-faster$thread_string --max-active=$max_active --beam=$beam --lattice-beam=$lattice_beam \
	    --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$srcdir/Graph/words.txt $decode_extra_opts \
	    $srcdir/final.mdl $srcdir/Graph/HCLG.fst "$feats" "ark:|gzip -c > $transdir/lat.JOB.gz" || exit 1;

	fi
	mv $transdir $lvcsrRootDir/trans
	cat $lvcsrRootDir/trans/decode_$fileRootName/log/decode.1.log | grep "^$fileRootName" | cut -d ' ' -f2- > $lvcsrRootDir/trans/decode_$fileRootName.log

        if [ "$indiceData" = "True" ] ; then
          confidence_score $lvcsrRootDir/trans/decode_$fileRootName.log $fileRootName
        fi

fi
	### for next sprint add
	### fmllr Feature Extraction ####
	### DNN Acoustic Models applied on top of the fmllr features ###
	### Rescoring with LM
	### Get CTM and STM files

#echo "End...."

