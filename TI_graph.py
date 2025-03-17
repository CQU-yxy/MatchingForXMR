import math
import time
from tkinter import _flatten

import networkx
import csv

import networkx as nx
import numpy as np
from networkx.algorithms import bipartite
import pandas as pd
from sklearn.utils import shuffle

import igraph as ig
from scipy.stats import gamma

'''this is the construction of TXOs-Input bipartite graph and 
the solution of full matching weight maxmum problem
1. recreate zero-mixin attack
2. organise data set and ground truth and prepare the time
3. construct graph with weight none, order-based, gamma-based, pareto-based
4. perform the matching and get the results, compute the accuracy'''

class TXOs_Input_graph:
    inputs = []
    txos = []
    edges = []
    ringsize = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]
                ,11:[],12:[],13:[],14:[],15:[]}
    ground_truth = {}
    path1 = "./csv-export/csv/input-output-refs.csv"
    path2 = "./csv-export/csv/test.csv"
    path3 = "./csv-export/csv/inputs.csv"
    path4 = "./csv-export/ground_truth/ringsize1.csv"
    path_data = "./csv-export/dataset/data.csv"
    path_gt = "./csv-export/dataset/dataset_gt.csv"
    path_input = './csv-export/dataset/input.csv'
    path_txos = './csv-export/dataset/txos.csv'
    path_edges = './csv-export/dataset/edges.csv'
    noneweight = './csv-export/dataset/noneweight.csv'
    gammaweight = './csv-export/dataset/gammaweight.csv'

    '''get inputs, txos ,edges'''
    def read_Data(self):
        with open(self.path2,'r') as f:
            read = csv.reader(f)
            for each in list(read):
                if each[0] not in self.inputs:
                    self.inputs.append(each[0])
                temp = each[1]
                # temp = temp.split('-')[1]
                if temp not in self.txos:
                    self.txos.append(temp)
                self.edges.append((temp,each[0]))
        f.close()

    # 这样太慢了 for循环的话 还是要从pandas入手emm
    def zero_mixin(self):
        # # 先按照ringsize分类吧
        # with open(self.path3,'r') as f:
        #     read = csv.reader(f)
        #     for each in list(read):
        #         if int(each[2]) > 15:
        #             continue
        #         else:
        #             self.ringsize[int(each[2])].append(each[0])
        # f.close()
        # df1 = pd.read_csv(self.path1,header=None)
        # df1.columns = ['input', 'txo']
        # print(df1)
        # other_mixins = []
        # for i in range(1,16):
        #     temp = self.ringsize[i]
        #     for j in temp:
        #         other_mixins.append(j)
        # print(other_mixins,len(other_mixins))
        # df2 =df1.loc[df1['input'].isin(other_mixins)]
        # df2.to_csv("./csv-export/csv/input-output-refs-1+.csv")

        df2 = pd.read_csv("./csv-export/csv/input-output-refs-1+.csv",usecols=['input','txo'])
        print(df2)

        groundtruth = pd.read_csv(self.path4)
        knowntxos = df2[df2['txo'].isin(groundtruth['txo'])]
        print(knowntxos)
        unknowntxos = df2.drop(knowntxos.index)
        print(unknowntxos)
        counts = unknowntxos['input'].value_counts()
        while counts.values.min() == 1:
            index = list(counts[counts.values==1].index)
            print(len(index))
            newgroundtruth = unknowntxos[unknowntxos['input'].isin(index)]
            print(newgroundtruth)
            groundtruth = pd.concat([groundtruth,newgroundtruth])
            knowntxos = df2[df2['txo'].isin(groundtruth['txo'])]
            print(knowntxos)
            unknowntxos = df2.drop(knowntxos.index)
            print(unknowntxos)
            counts = unknowntxos['input'].value_counts()
            print(counts)
        groundtruth.to_csv('./csv-export/ground_truth/ringsize1+.csv',index=0)

        # with open(self.path1, 'r') as f:
        #     zero_mixin = self.ringsize[0]
        #     print(zero_mixin,len(zero_mixin))
        #     print(time.time())
        #     read = csv.reader(f)
        #     print(time.time())
        #     for each in read:
        #         for zm in zero_mixin:
        #             t1 = int(str(zm).split('i')[1])
        #             t0 = int(each[0].split('i')[1])
        #             if t0 > t1:
        #                 continue
        #             elif t0 == t1:
        #                 self.ground_truth[each[0]] = each[1]
        #                 zero_mixin.remove(zm)
        #                 print(zm)
        #             else:
        #                 break
        #     print(time.time())
        #     f.close()
        # print("finish2")
        # with open("./csv-export/ground_truth/ringsize1.csv",'w') as f1:
        #     write = csv.writer(f1)
        #     for key,value in self.ground_truth.items():
        #         write.writerow([key,value])



        # 得到了一个字典，key:ringsize；value: txs


    def get_dataset(self):
        # '''get ground truth for dataset'''
        df1 = pd.read_csv(self.path4,usecols=['input','txo'])
        df2 = pd.read_csv('./csv-export/ground_truth/ringsize1+.csv',usecols=['input','txo'])
        df3 = df2.drop(df2[df2['input'].isin(df1['input'])].index)
        print(df3)

        df3.to_csv('./csv-export/dataset/dataset_gt.csv', index=0)

        '''txs data for dataset'''
        gt = pd.read_csv('./csv-export/dataset/dataset_gt.csv',usecols=['input','txo'])
        total_data = pd.read_csv('./csv-export/csv/input-output-refs-1+.csv',usecols=['input','txo'])
        dataset = total_data[total_data['input'].isin(gt['input'])]
        print(dataset)
        dataset.to_csv('./csv-export/dataset/data.csv',index=0)

        '''prepare time for input'''
        df1 = pd.read_csv("./csv-export/csv/input-rels.csv",header=None)
        df1.columns = ['tx', 'input']
        print(df1)

        df2 = pd.read_csv(self.path_data,usecols=['input','txo'])
        print(df2)
        df3 = pd.merge(df1,df2,on='input')
        print(df3)

        df4 = pd.read_csv("./csv-export/csv/tx-blocks.csv",header=None)
        df4.columns = ['tx','block']
        print(df4)
        df5 = pd.merge(df3,df4,on='tx')
        print(df5)

        df6 = pd.read_csv('./csv-export/csv/blocks.csv',header=None)
        df6.columns = ['block', 'temp1', 'hash', 'timestamp']
        df6 = df6[['block','timestamp']]
        print(df6)
        df = pd.merge(df5,df6,on='block')
        print(df)
        df.to_csv('./csv-export/dataset/data_with_time.csv',index=0)

        '''prepare time for txos'''
        df0 = pd.read_csv('./csv-export/dataset/data_with_time.csv',)
        df0.columns = ['itx','input','txo','iblock','itimestamp']

        df1 = pd.read_csv("./csv-export/csv/output-rels.csv", header=None)
        df1.columns = ['otx', 'txo']
        print(df1)

        df2 = pd.merge(df0,df1,on='txo')
        print(df2)
        df3 = pd.read_csv("./csv-export/csv/tx-blocks.csv", header=None)
        df3.columns = ['otx', 'oblock']
        print(df3)
        df4 = pd.merge(df2, df3, on='otx')
        print(df4)

        df5 = pd.read_csv('./csv-export/csv/blocks.csv', header=None)
        df5.columns = ['oblock', 'temp1', 'hash', 'otimestamp']
        df5 = df5[['oblock', 'otimestamp']]
        print(df5)
        df = pd.merge(df4, df5, on='oblock')
        print(df)
        df.to_csv('./csv-export/dataset/data_with_time.csv', index=0)

        '''compute the age of txo'''
        df = pd.read_csv('./csv-export/dataset/data_with_time.csv')
        df['age'] = df['itimestamp'] - df['otimestamp']
        df.to_csv('./csv-export/dataset/data_with_time.csv', index=0)

        df = pd.read_csv('./csv-export/dataset/data_with_time.csv')
        df_input = df[['input']]
        df_input = df_input.drop_duplicates(subset='input')
        print(df_input,df_input['input'].value_counts())
        df_input.to_csv(self.path_input,index=0)

        df_txos = df[['txo']]
        print(df_txos)
        df_txos.to_csv(self.path_txos, index=0)

        df_edges = df[['input','txo','age']]
        print(df_edges)
        df_edges.to_csv(self.path_edges, index=0)




    def test(self):
        # t = pd.read_csv(self.path1)
        df = pd.read_csv('csv-export/dataset/gammaweight.csv')
        print(df)
        df.columns = ['input','txo','weight']
        df.to_csv('./gammaweight.csv',index=0)


    '''networkx速度太慢了，改用igraph
    igraph和一般的networkx的写法不太一样 更麻烦 需要转换'''
    def graph_construct(self):
        # '''read data'''
        # self.inputs = _flatten(pd.read_csv(self.path_input).values.tolist())
        # self.txos = _flatten(pd.read_csv(self.path_txos).values.tolist())
        # edges = pd.read_csv(self.path_edges).values.tolist()
        # for i in edges:
        #     self.edges.append(tuple(i))
        #
        # '''none weight'''
        # B = nx.Graph()
        # B.add_nodes_from(self.inputs,bipartite=0)
        # B.add_nodes_from(self.txos, bipartite=1)
        # B.add_weighted_edges_from(self.edges)
        # match = bipartite.maximum_matching(B,top_nodes=self.inputs)

        # df = pd.read_csv(self.path_edges)
        # df2 = df.sort_values(by=['input'])
        # df2['age'] = 0
        # df2.columns = ['input','txo','weight']
        # df2.to_csv('./csv-export/dataset/noneweight.csv',index=0)

        '''for none weight'''
        edges = pd.read_csv('./newdataset/gammaweight/gamma8.csv')
        edges = edges[['input','txo']]
        edges = shuffle(edges)
        print(edges)
        print(time.time())
        g = ig.Graph.DataFrame(edges,directed=False,use_vids=False)
        print(time.time())
        print(g.vcount(),g.ecount())
        print(g.is_bipartite())
        types = []
        transfer = {}
        for v in g.vs['name']:
            if 'i' in v:
                types.append(0)
            elif '-' in v:
                types.append(1)
            else:
                print('error!!!!')
        for j in range(0,len(g.vs)):
            transfer[j] = g.vs[j]['name']
        none_matching = g.maximum_bipartite_matching(types=types)
        print(none_matching)
        results = []

        for j in none_matching.edges():
            (x,y) = j.tuple
            if 'i' in transfer[x]:
                results.append([transfer[x],transfer[y]])
            else:
                results.append([transfer[y], transfer[x]])
        print(results,len(results))
        df = pd.DataFrame(results)
        df.to_csv('./newdataset/noneweight/result/none8.csv', index=0)
        # df.to_csv('./csv-export/results/none52.csv',index=0)

        '''for gamma weight'''
        # df = pd.read_csv('./orderweight/ordertotal.csv')
        # edges = df.loc[0:80000]
        # edges.to_csv('./test.csv')
        # edges = pd.read_csv('./test.csv')
        # edges = edges[['input', 'txo', 'weight']]
        # print(time.time())
        # print(edges)
        # weights = edges['weight'].values.tolist()
        # print(len(weights))
        # g = ig.Graph.DataFrame(edges, directed=False, use_vids=False)
        # print('bipartite graph',g.is_bipartite())
        # types = []
        # transfer = {}
        # results = []
        # for v in g.vs['name']:
        #     if 'i' in v:
        #         types.append(0)
        #     elif '-' in v:
        #         types.append(1)
        #     else:
        #         print('error!!!!')
        # for j in range(0, len(g.vs)):
        #     transfer[j] = g.vs[j]['name']
        # print(time.time())
        # gamma_matching = g.maximum_bipartite_matching(types=types,weights=weights)
        # print(time.time())
        # for j in gamma_matching.edges():
        #     (x,y) = j.tuple
        #     results.append([transfer[x],transfer[y]])
        # print(results)
        # df = pd.DataFrame(results)
        # df.columns = ['input','txo']
        # print(df)
        # df.to_csv('./test2.csv',index=0)

    def MWFM_to_maxflow(self):
        edges = pd.read_csv('./noneweight/none1.csv')
        edges = edges[['input', 'txo', 'weight']]
        edges = shuffle(edges)

        print(edges['input'].drop_duplicates())
        print(time.time())
        print(edges)

        weights = edges['weight'].values.tolist()
        weights = list(map(lambda x:1, weights))
        g = ig.Graph.DataFrame(edges, directed=True, use_vids=False)
        transfer = {}
        print(g.vcount())
        # g.add_vertices(2)
        g.add_vertex(name='0')
        g.add_vertex(name='1')
        source = len(g.vs['name']) - 2
        target = source + 1
        types = []
        results = []
        temp = []
        count = 0
        print(source, target)
        addedge1 = []
        addedge2 = []
        for j in range(0, len(g.vs)):
            transfer[j] = g.vs[j]['name']
        for v in g.vs['name']:
            if 'i' in v:
                types.append(0)
                addedge1.append((source,v))
                # g.add_edges([(source,v)])
                temp.append(1)
                count+=1
            elif '-' in v:
                types.append(1)
                addedge2.append((v, target))
                # g.add_edges([(v, target)])
                temp.append(1)
            else:
                print('error!!!!')
        print(time.time())
        g.add_edges(addedge1)
        g.add_edges(addedge2)
        weights = weights + temp
        maxflow = g.maxflow(source=source,target=target,capacity=weights)
        print(maxflow)
        print(len(maxflow.flow))
        candidate = []
        for j in range(0,len(maxflow.flow)):
            if maxflow.flow[j] >= 0.6:
                (x,y) = g.es[j].tuple
                if x != source and y != target:
                    results.append([transfer[x], transfer[y]])
        # for j in range(0,len(maxflow.flow)):
        #     if maxflow.flow[j] > 0 and maxflow.flow[j] < 0.8:
        #         (x, y) = g.es[j].tuple
        #         if x != source and y != target:
        #             candidate.append([transfer[x], transfer[y]])
        print(len(candidate))
        df1 = pd.DataFrame(results)
        df1.columns = ['input', 'txo']
        flatresults = list(_flatten(results))
        print(df1)
        # for i in candidate:
        #     print(i)
        #     if i[0] in flatresults or i[1] in flatresults:
        #         continue
        #     else:
        #         results.append(i)
        #         flatresults.append(i[0])
        #         flatresults.append(i[1])
            # (x, y) = j.tuple
            # print(x,y)
            # # if y != target and x != source:
            # results.append([transfer[x], transfer[y]])
        df = pd.DataFrame(results)
        df.columns = ['input', 'txo']
        print(df)
        print(df['txo'].value_counts())
        print(df['input'].value_counts())
        print(time.time())
        df.to_csv('./results/ordertotal.csv',index=0)




    def compute_weight(self):
        '''gamma-based'''
        edges = pd.read_csv(self.path_edges)

        input_group = {}
        count = 0
        for edge in edges.itertuples():
            # 1: input ; 2 txo; 3 age
            temp = [edge[2],edge[3]]
            count += 1
            print(count)
            if edge[1] in input_group.keys():
                input_group[edge[1]].append(temp)
            else:
                input_group[edge[1]] = [temp]
        print(len(input_group))

        for input in input_group.keys():
            temp = input_group[input]
            ages = []
            for i in range(0,len(temp)):
                ages.append(temp[i][1])
            weights = self.compute_order(ages)
            # weights = self.normalization(prob)
            for i in range(0,len(temp)):
                temp[i][1] = weights[i]
        print(input_group)
        final = []
        for input in input_group.keys():
            temp = input_group[input]
            for i in range(0, len(temp)):
                final.append([input,temp[i][0],temp[i][1]])
        df = pd.DataFrame(final)
        df.columns = ['input','txo','weight']
        print(df)
        df.to_csv('./ordernew-1w.csv',index=0)


    def compute_gamma(self,ages):
        logages = []
        for i in ages:
            print(i)
            if i > 0:
                logages.append(math.log(i))
            else:
                logages.append(100)
        gamma_prob = gamma.pdf(logages,a=19.28,scale=1/1.61)
        return gamma_prob

    def normalization(self,prob):
        minvalue = min(prob)
        maxvalue = max(prob)
        weights = []
        for i in prob:
            if maxvalue == minvalue:
                weights.append(1)
            else:
                weights.append((i-minvalue)/(maxvalue-minvalue))
        return weights

    def compute_pareto(self,ages):
        pareto_prob = []
        for i in ages:
            print(i)
            if i > 0:
                logi = math.log(i)
                temp = pow((logi + 1), -1.172)
                pareto_prob.append(temp)
            else:
                logi = 100
                temp = pow((logi + 1), -1.172)
                pareto_prob.append(temp)
        return pareto_prob

    def compute_order(self,ages):
        weights = []
        ages.sort(reverse=True)
        for i in range(0,len(ages)):
            temp =  (i/(len(ages)-1))
            weights.append(temp)
        return weights

    def compute_accuracy(self):
        df1 = pd.read_csv('./results/ordertotal.csv')
        df1.columns = ['input','txo']
        # ground truth
        df2 = pd.read_csv('./groundtruth/gttotal.csv')

        # 取交集
        df3 = pd.merge(df1,df2,how ='inner')
        rightnum = len(df3)
        totalnum = len(df1)
        print(df1)
        print(df2)
        print(df3)

        precision = rightnum / len(df1)
        recall = rightnum / len(df2)
        f1score = 2*precision*recall / (precision+recall)
        print(precision,recall,f1score)
        return precision, recall, f1score

    def help(self):
        df = pd.read_csv('./csv-export/dataset/gammaweight.csv')
        df = df[0:100000]
        df.to_csv('./test.csv',index = 0)

    def split_dataset(self):
        # source = pd.read_csv('./csv-export/dataset/noneweight.csv')
        # for row in source.itertuples():
        #     if 'i' not in row[1]:
        #         print(row[1])
        df1 = pd.read_csv('./csv-export/dataset/data.csv')
        df2 = pd.read_csv('./ordernew.csv')
        df3 = pd.merge(df1,df2,how='inner',on=['input','txo'])
        # print(len(df1),len(df2),len(df3))
        print(df1,df2,df3)
        df3.to_csv('./orderweight/ordertotal.csv')
        # df4 = pd.concat([df3,df2,df2]).drop_duplicates(keep=False)

        total = pd.read_csv('./orderweight/ordertotal.csv')
        df1 = total[0:1865707]
        df2 = total[1865707:2694428]
        df3 = total[2694428:4399116]
        df4 = total[4399116:7174295]
        df5 = total[7174295:13289475]
        df6 = total[13289475:]

        df1.to_csv('./orderweight/order1.csv',index=0)
        df2.to_csv('./orderweight/order2.csv', index=0)
        df3.to_csv('./orderweight/order3.csv', index=0)
        df4.to_csv('./orderweight/order4.csv', index=0)
        df5.to_csv('./orderweight/order5.csv', index=0)
        df6.to_csv('./orderweight/order6.csv', index=0)

        df = pd.read_csv('./csv-export/dataset/dataset_gt.csv')
        # df.to_csv('./groundtruth/gttotal.csv',index=0)
        # df1 = df[0:502652]
        # df3 = df[502652:744108]
        # df4 = df[744108:1061962]
        # df5 = df[1061962:1455809]
        # df6 = df[1455809:3661523]
        # df2 = df[3661523:]
        #
        # df1.to_csv('./groundtruth/gt1.csv',index=0)
        # df2.to_csv('./groundtruth/gt6.csv', index=0)
        # df3.to_csv('./groundtruth/gt2.csv', index=0)
        # df4.to_csv('./groundtruth/gt3.csv', index=0)
        # df5.to_csv('./groundtruth/gt4.csv', index=0)
        # df6.to_csv('./groundtruth/gt5.csv', index=0)

    def test_gt(self):
        df1 = pd.read_csv('./groundtruth/')

    def newsplit(self):
        df = pd.read_csv('./groundtruth/gttotal.csv')
        # ground truth就不分了，到时候直接取交集除就好了

        df =pd.read_csv('./newdataset/paretoweight/paretototal.csv')
        df8 = df[13289475:]
        df7 = df[9574566:13289475]
        df6 = df[7174295:9574566]
        df5 = df[5002126:7174295]
        df4 = df[4399116:5002126]
        df3 = df[3382423:4399116]
        df2 = df[2694428:3382423]
        df1 = df[2099480:2694428]
        print(df1,df2,df3,df4,df5,df6,df7,df8)

        df1.to_csv('./newdataset/paretoweight/pareto1.csv', index=0)
        df2.to_csv('./newdataset/paretoweight/pareto2.csv', index=0)
        df3.to_csv('./newdataset/paretoweight/pareto3.csv', index=0)
        df4.to_csv('./newdataset/paretoweight/pareto4.csv', index=0)
        df5.to_csv('./newdataset/paretoweight/pareto5.csv', index=0)
        df6.to_csv('./newdataset/paretoweight/pareto6.csv', index=0)
        df7.to_csv('./newdataset/paretoweight/pareto7.csv', index=0)
        df8.to_csv('./newdataset/paretoweight/pareto8.csv', index=0)


    def newest(self):
        df = pd.read_csv('./newdataset/orderweight/order8.csv')

        df = df[['input','txo','weight']]
        df = df.loc[df['weight']==1]
        print(df)

        df = df[['input','txo']]
        print(df)
        df.to_csv('./newdataset/newest/newest8.csv',index=0)

    def compute_acc(self):

        df1 = pd.read_csv('./newdataset/newest/newest1.csv')
        df1.columns = ['input', 'txo']
        # ground truth
        df2 = pd.read_csv('./groundtruth/gttotal.csv')

        # 取交集
        df3 = pd.merge(df1, df2, how='inner')
        rightnum = len(df3)
        totalnum = len(df1)
        print(df1)
        print(df2)
        print(df3)

        precision = rightnum / len(df1)
        recall = rightnum / len(df2)
        f1score = 2 * precision * recall / (precision + recall)
        print(precision, recall, f1score)
        return precision, recall, f1score

def statistic_snapshot():
    df1 = pd.read_csv('./newdataset/paretoweight/pareto8.csv')
    df1 = df1[['input', 'txo']]
    print(df1.nunique())









if __name__ == '__main__':
    '主函数调用'
    graph = TXOs_Input_graph()
    # graph.graph_construct()
    # graph.MWFM_to_maxflow()
    # graph.zero_mixin()
    # graph.get_dataset()
    # merge_csv()
    # graph.test()
    # graph.compute_weight()
    # graph.help()
    graph.compute_accuracy()
    # graph.split_dataset()
    # graph.newsplit()
    # graph.newest()
    # graph.compute_acc()
    # statistic_snapshot()

