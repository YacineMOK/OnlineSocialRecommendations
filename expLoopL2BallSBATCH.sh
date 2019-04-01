#!/bin/bash
PREFIX=./Experiments/
LOGPREFIX=./Logs/ 
PREFIX=./
iter=100
pyth=python3

for n in 2 10 100 500
do
  for d in 2 5 10 20
  do
      for sgm in 0.1 1 2
      do
        for sc in 1 0.1 0.01 0.001 0.0001 0.00001 0.000001
        do
          for net in cmp erdos-renyi barabasi-albert
          do
            for str in LinOptL2Ball RandomBanditL2Ball LinREL1L2Ball\
               LinUCB RegressionL2Ball 
            do
              echo "Launching $str n=${n} d=${d} sigma=${sgm} scale=${sc}"
              fname="${str}_${net}_n${n}_d${d}_sigma${sgm}_scale${sc}"
              sbatch --partition=$1 SocialBandits.sbatch $str --n $n --d $d --sigma ${sgm}\
                --maxiter ${iter} --scale ${sc} --graphtype ${net}\
                --outfile "$PREFIX${fname}.csv" --logfile\
                "$LOGPREFIX${fname}.log" --noscreenoutput

              sbatch --partition=$1 SocialBandits.sbatch $str --n $n --d $d --sigma ${sgm}\
                --maxiter ${iter} --scale ${sc} --graphtype ${net}\
                --stochastic True --outfile "$PREFIX${fname}_stochastic.csv"\
                --logfile "$LOGPREFIX${fname}_stochastic.log" --noscreenoutput
              sbatch --partition=$1 SocialBandits.sbatch $str --n $n --d $d --sigma ${sgm}\
                --maxiter ${iter} --scale ${sc} --graphtype ${net}\
                --infinite True --outfile "$PREFIX${fname}_infinite.csv"\
                --logfile "$LOGPREFIX${fname}_infinite.log" --noscreenoutput
            done
          done
        done
      done
    done
  done
