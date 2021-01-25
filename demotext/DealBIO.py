# -*- coding: UTF-8 -*-
__author__ = 'zy'
__time__ = '2020/10/26 15:32'
import re
# def deal_(file):#seadas oceancolor
#     with open(file,'r') as f:
#         tmp=f.read()
#     #进行正则抽取 L1A IDL
#     with open(file+'naonao','w+') as f:
#         for i in tmp:
#             #替换句号为seq
#         #[@先生#Location*]
#         Bre='[@.*?#Location*]'
#         res = re.findall(Bre, '阅读数为2 点赞数为10')
#         print(res)
#         f.write()
a=[]
file_list=['2016.txt.anns']
with open('2016.txt','w+',encoding='utf-8') as f2:
    for j in file_list:
        with open(j,'r+',encoding='gbk') as f:#,encoding='utf-8'
            tmp=f.read()
        ss=tmp.split('\n')
        print(ss)
        for i in ss:
           labels=i.split("\t")#\t
           try:
               if labels[1]!='O':
                   index=0
                   for j in labels[0]:
                       if index!=0:
                           a.append(j+' '+labels[1].replace('B-','I-'))
                       else:
                           a.append(j+' '+labels[1])
                       index=index+1
               else:
                   index = 0
                   for j in labels[0]:
                       a.append(j+' '+labels[1])
           except:
               pass
        print(a)
        #处理逗号

        for i in a:
            # print(i)
            f2.write(i)
            f2.write('\n')
            if i=="。 O":
                f2.write('\n')