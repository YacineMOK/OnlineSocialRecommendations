#!/bin/bash
n=840
d=28
dset=flixstr
PREFIX=./
iter=100
pyth=python3

    for M in 100
    do
      for sgm in 1
      do
        for sc in 0.00001
        do
            for str in LinREL1FiniteSet\
               LinREL2FiniteSet RegressionFiniteSet 
            do
              echo "$str n=${n} d=${d} M=${M} sigma=${sgm} scale=${sc}"
              fname="${str}_${dset}_n${n}_d${d}_M${M}_sigma${sgm}_scale${sc}"
              $pyth SocialBandits.py $str --n $n --d $d --M $M --sigma ${sgm}\
                --maxiter ${iter} --scale ${sc} --u0file "${dset}U0.csv"\
                --networkfile "${dset}Links.csv" --outfile "$PREFIX${fname}.csv"
              $pyth SocialBandits.py $str --n $n --d $d --M $M --sigma ${sgm}\
                --maxiter ${iter} --scale ${sc} --u0file "${dset}U0.csv"\
                --outfile "$PREFIX${fname}_nonet.csv"
            done
          done
        done
      done
