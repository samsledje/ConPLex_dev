MODEL=$1
OUTDIR=$2
mkdir -p $OUTDIR

for i in $(cat ./dataset/DUDe/dude_within_type_train_test_split.csv | grep test | awk -F',' '{print $1}');
do
    echo $i;
    python DUDE_evaluate_decoys.py --model $MODEL --outdir $OUTDIR $i
done

python DUDE_summarize_decoys.py $OUTDIR > $OUTDIR/summary.txt
