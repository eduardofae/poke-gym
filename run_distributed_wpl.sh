export PYTHONPATH=$(pwd)

process_number=12
count=`expr $process_number - 1`

for i in `seq 0 $count`
do
  uv run Trainer/Deep/Test/Distributed/TestDistributedDeepWPL.py "$i" "$process_number" &
  echo uv run TestDistributedDeepWPL.py "$i" "$process_number"
  sleep 1
done
wait
echo Done!
