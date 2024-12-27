containers=("[37,26,13]" "[38,26,13]" "[40,28,16]" "[42,30,18]" "[42,30,40]" "[52,40,17]" "[54,45,36]" "[35,23,13]")

for container in "${containers[@]}"; do
    echo $container
    python main.py --continuous --setting 2 --container-size "$container" --total-updates 7000 --num-processes 80 | tee "logger/${container}.txt"
done
