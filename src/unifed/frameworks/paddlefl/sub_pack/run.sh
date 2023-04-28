source /root/.bashrc

python3.8 download.py 
python3.8 leaf_utils.py

unset http_proxy
unset https_proxy
ps -ef | grep -E fl_ | grep -v grep | awk '{print $2}' | xargs kill -9

mkdir logs

python3.8 fl_master.py > logs/master.log 2>&1 &
sleep 3
python3.8 -u fl_scheduler.py > logs/scheduler.log 2>&1 &
sleep 5
python3.8 -u fl_server.py > logs/server0.log 2>&1 &
sleep 2

source tot_num
for ((i=0;i< $tot_num ;i++))
do
    python3.8 -u fl_trainer.py $i > logs/trainer$i.log 2>&1 &
done


while ((1))
do
    source if_end
    
    if [[ $if_end = 'true' ]]
    then break
    fi

    sleep 1
done

/bin/bash stop.sh

# python convert_metric.py