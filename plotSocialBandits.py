#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib
#not using the X display
matplotlib.use('Agg',force=True)
import matplotlib.pyplot as plt


yaxis_defs = {'time':'Time (sec)', 'reward':'Regret'}

finite_methods = ['RandomBanditFiniteSet', 'LinREL1FiniteSet',\
        'LinREL2FiniteSet', 'RegressionFiniteSet']
finite_best = 'LinOptFiniteSet'

l2ball_methods = ['RandomBanditL2Ball', 'LinREL1L2Ball',\
        'RegressionL2Ball']
l2ball_best = 'LinOptL2Ball'

axis = ['time','reward']
ns = [2,10]
ds = [2,5,10]
Ms = [10,100,1000]
sgms = [0.1, 1, 2]
scs = ['1','0.1','0.01','0.001','0.0001','0.00001','0.000001']

nets = ['cmp','erdos-renyi','barabasi-albert']
suffixs = ['','_infinite','_stochastic']

prefix = './experiments/'


def plotFiniteSet(axis='reward', n=2, d=2, M=10, sgm=0.1, sc=1, net='cmp',\
        prefix='', suffix='', cumm=False):
    val = 0.0
    pltname = '%s%s_%s_n%s_d%s_M%s_sigma%s_scale%s%s.pdf'%\
            (prefix, axis, net, n, d, M, sgm, sc, suffix)
    xopt = []
    if axis == 'reward':
        fname = '%s%s_%s_n%s_d%s_M%s_sigma%s_scale%s%s.csv'%\
                (prefix, finite_best, net, n, d, M, sgm, sc, suffix)
        df = pd.read_csv(fname)
        xopt = df[axis]
 
    for met in finite_methods:
        fname = '%s%s_%s_n%s_d%s_M%s_sigma%s_scale%s%s.csv'%\
                (prefix, met, net, n, d, M, sgm, sc, suffix)
        df = pd.read_csv(fname)
        if axis=='reward':
            xvals = np.cumsum([abs(x[0]-x[1]) for x in zip(df[axis],xopt)])
        elif axis=='time':
            xvals = np.cumsum(df[axis])
        else:
            xvals = df[axis]
        plt.plot(xvals, label=met)

    plt.xlabel('Step')
    plt.ylabel(yaxis_defs[axis])
    plt.legend(loc='upper left')
    print pltname
    plt.savefig(pltname, format='pdf')
    plt.clf()

def plotL2Ball(axis='reward', n=2, d=2, sgm=0.1, sc=1, net='cmp',\
        prefix='', suffix='', cumm=False):
    val = 0.0
    pltname = '%s%s_%s_n%s_d%s_l2ball_sigma%s_scale%s%s.pdf'%\
            (prefix, axis, net, n, d, sgm, sc, suffix)
    xopt = []
    if axis == 'reward':
        fname = '%s%s_%s_n%s_d%s_sigma%s_scale%s%s.csv'%\
                (prefix, l2ball_best, net, n, d, sgm, sc, suffix)
        df = pd.read_csv(fname)
        xopt = df[axis]
 
    for met in l2ball_methods:
        fname = '%s%s_%s_n%s_d%s_sigma%s_scale%s%s.csv'%\
                (prefix, met, net, n, d, sgm, sc, suffix)
        df = pd.read_csv(fname)
        if axis=='reward':
            clean_vals = []
            for x in df[axis]:
                val = x
                if type(x) is str:
                    val = val[1:-1]
                    clean_vals.append(complex(val))
                else:
                    clean_vals.append(val)
            xvals = np.cumsum([abs(x[0].real-x[1].real) for x in\
                    zip(clean_vals,xopt)])
        elif axis=='time':
            xvals = np.cumsum(df[axis])
        else:
            xvals = df[axis]
        plt.plot(xvals, label=met)

    plt.xlabel('Step')
    plt.ylabel(yaxis_defs[axis])
    plt.legend(loc='upper left')
    print pltname
    plt.savefig(pltname, format='pdf')
    plt.clf()

#plot comparisons of parameters for each algorithm
def plotBarChartAlgo(prefix, suffix, opt, algos, fmt, plname, val_name,\
        val_arr, axis='reward'):
    ind = np.arange(len(val_arr))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pltname='%s%s_%s_%s%s.pdf'%(prefix,axis,val_name,plname,suffix)
    print pltname
    i = 0
    w = 0.2
    for met in algos:
        print met
        val_hist = []
        for val in val_arr:
            #optimal
            fname = '%s%s_%s%s.csv'%(prefix,opt,fmt%val,suffix)
            df_opt = pd.read_csv(fname)
            #method
            fname = '%s%s_%s%s.csv'%(prefix,met,fmt%val,suffix)
            df = pd.read_csv(fname)
            clean_vals = []
            for x in df[axis]:
                val = x
                if type(x) is str:
                    val = val[1:-1]
                    clean_vals.append(complex(val))
                else:
                    clean_vals.append(val) 
            val_hist.append(np.cumsum([abs(x[0].real-x[1].real) for x in\
                    zip(clean_vals,df_opt[axis])])[-1:][0])
        print val_hist
        ax.bar(ind+i*w,val_hist,w,label=met)
        i += 1
    ax.set_ylabel(yaxis_defs[axis])
    ax.set_xticks(ind)
    ax.set_xticklabels(val_arr)
    ax.legend()
    plt.savefig(pltname, format='pdf')
    plt.clf()

if __name__=='__main__':
    '''
    for ax in axis:
        for n in ns:
            for d in ds:
                for M in Ms:
                    for sgm in sgms:
                        for sc in scs:
                            for net in nets:
                                for suffix in suffixs:
                                    plotFiniteSet(ax, n, d, M, sgm, sc, net,\
                                         prefix, suffix)
    
    for ax in axis:
        for n in ns:
            for d in ds:
                for sgm in sgms:
                    for sc in scs:
                        for net in nets:
                            for suffix in suffixs:
                                plotL2Ball(ax, n, d, sgm, sc, net, prefix,\
                                        suffix)
    
    #plots for FiniteSet and parameter comparison
    sca_fmt = 'cmp_n10_d5_M100_sigma1_scale%s'
    sca_plname = 'cmp_n10_d5_M100_sigma1'
    sca_name = 'scale'
    sca_arr = ['1','0.1','0.01','0.001','0.0001','0.00001','0.000001']
    sgm_fmt = 'cmp_n10_d5_M100_sigma%s_scale0.00001'
    sgm_plname = 'cmp_n10_d5_M100_scale0.00001'
    sgm_name = 'sigma'
    sgm_arr = ['2','1','0.1'] 
    M_fmt = 'cmp_n10_d5_M%s_sigma1_scale0.00001'
    M_plname = 'cmp_n10_d5_sigma1_scale0.00001'
    M_name = 'items'
    M_arr = ['10','100','1000']
    tp_fmt = 'cmp_n10_d5_M100_sigma1_scale0.00001%s'
    tp_plname ='cmp_n10_d5_M100_sigma1_scale0.00001'
    tp_name = 'types'
    tp_arr = ['','_infinite','_stochastic']
    net_fmt = '%s_n100_d5_M1000_sigma1_scale0.000001'
    net_plname ='n100_d5_M1000_sigma1_scale0.000001'
    net_name = 'nets'
    net_arr = ['cmp','erdos-renyi','barabasi-albert']
    opt='LinOptFiniteSet'
    algos=['LinREL1FiniteSet','LinREL2FiniteSet','RegressionFiniteSet']
    plotBarChartAlgo(prefix, '', opt, algos, sca_fmt, sca_plname, sca_name, sca_arr)
    plotBarChartAlgo(prefix, '', opt, algos, sgm_fmt, sgm_plname, sgm_name, sgm_arr)
    plotBarChartAlgo(prefix, '', opt, algos, M_fmt, M_plname, M_name, M_arr)
    plotBarChartAlgo(prefix, '', opt, algos, tp_fmt, tp_plname, tp_name, tp_arr) 
    plotBarChartAlgo(prefix, '', opt, algos, net_fmt, net_plname, net_name, net_arr) 
    #plots for L2Ball and parameter comparison
    sca_fmt = 'cmp_n10_d5_sigma1_scale%s'
    sca_plname = 'cmp_n10_d5_l2ball_sigma1'
    sca_name = 'scale'
    sca_arr = ['1','0.1','0.01','0.001','0.0001','0.00001','0.000001']
    sgm_fmt = 'cmp_n10_d5_sigma%s_scale0.00001'
    sgm_plname = 'cmp_n10_d5_l2ball_scale0.00001'
    sgm_name = 'sigma'
    sgm_arr = ['2','1','0.1'] 
    tp_fmt = 'cmp_n10_d5_sigma1_scale0.00001%s'
    tp_plname ='cmp_n10_d5_l2ball_sigma1_scale0.00001'
    tp_name = 'types'
    tp_arr = ['','_infinite','_stochastic']
    net_fmt = '%s_n100_d5_sigma1_scale0.000001'
    net_plname ='n100_d5_l2ball_sigma1_scale0.000001'
    net_name = 'nets'
    net_arr = ['cmp','erdos-renyi','barabasi-albert'] 
    opt='LinOptL2Ball'
    algos=['LinREL1L2Ball','RegressionL2Ball']
    plotBarChartAlgo(prefix, '', opt, algos, sca_fmt, sca_plname, sca_name, sca_arr)
    plotBarChartAlgo(prefix, '', opt, algos, sgm_fmt, sgm_plname, sgm_name, sgm_arr)
    plotBarChartAlgo(prefix, '', opt, algos, tp_fmt, tp_plname, tp_name, tp_arr)
    plotBarChartAlgo(prefix, '', opt, algos, net_fmt, net_plname, net_name, net_arr) 
    '''

    d_arr = ['2','5']
    l2ball_methods = ['RandomBanditL2Ball', 'LinREL1L2Ball','RegressionL2Ball',\
        'LinUCB']
    l2ball_best = 'LinOptL2Ball'
    for ax in ['time','reward']:
        for d in d_arr:
            plotL2Ball(ax, '10', d, 1, '0.00001', 'cmp', prefix, '')
 
