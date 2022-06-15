MODEL=$1
OUTDIR=$2

for i in $(cat ./nbdata/dude_within_type_train_test_split.csv | grep test | awk -F',' '{print $1}');
do
    echo $i;
    python evalute_decoys.py --model $MODEL --outdir $OUTDIR $i
done

python summarize_decoys.py $OUTDIR > $OUTDIR/summary.txt
