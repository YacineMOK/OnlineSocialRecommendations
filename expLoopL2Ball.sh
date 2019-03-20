#!/bin/bash
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
              echo "$str n=${n} d=${d} sigma=${sgm} scale=${sc}"
              fname="${str}_${net}_n${n}_d${d}_sigma${sgm}_scale${sc}"
              $pyth SocialBandits.py $str --n $n --d $d --sigma ${sgm}\
                --maxiter ${iter} --scale ${sc} --graphtype ${net}\
                --outfile "$PREFIX${fname}.csv"
              $pyth SocialBandits.py $str --n $n --d $d --sigma ${sgm}\
                --maxiter ${iter} --scale ${sc} --graphtype ${net}\
                --stochastic True --outfile "$PREFIX${fname}_stochastic.csv"
              $pyth SocialBandits.py $str --n $n --d $d --sigma ${sgm}\
                --maxiter ${iter} --scale ${sc} --graphtype ${net}\
                --infinite True --outfile "$PREFIX${fname}_infinite.csv"
 
            done
          done
        done
      done
    done
  done
