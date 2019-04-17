#!/bin/bash
PREFIX=./Experiments/
LOGPREFIX=./Logs/ 
iter=100
pyth=python3

for n in 100 500
do
  for d in 2 5 10 20
  do
    for M in 10 100 1000
    do
      for sgm in 0.1 1 2
      do
        for sc in 1 0.1 0.01 0.001 0.0001 0.00001 0.000001
        do
          for net in cmp erdos-renyi barabasi-albert
          do
            for str in LinOptFiniteSet RandomBanditFiniteSet LinREL1FiniteSet\
               LinREL2FiniteSet RegressionFiniteSet 
            do
              echo "Launching $str n=${n} d=${d} M=${M} sigma=${sgm} scale=${sc}"
              fname="${str}_${net}_n${n}_d${d}_M${M}_sigma${sgm}_scale${sc}"
              sbatch --partition=$1 SocialBandits.sbatch $str --n $n --d $d --M $M --sigma ${sgm}\
                --maxiter ${iter} --scale ${sc} --graphtype ${net}\
                --outfile "$PREFIX${fname}.csv" --logfile "$LOGPREFIX${fname}.log" --noscreenoutput
            done
          done
        done
      done
    done
  done
done
