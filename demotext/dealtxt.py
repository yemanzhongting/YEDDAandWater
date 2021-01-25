# -*- coding: UTF-8 -*-
__author__ = 'zy'
__time__ = '2020/11/16 21:47'
# with open('1.txt','r+',encoding='utf-8') as f:
#     tmp=f.read()
#
# print(tmp)
#
# result=[]
#
# for i in tmp:
#     if i!='\t' and i!=' ':
#         result.append(i)
#
# with open('juzi.txt','w+',encoding='utf-8') as f:
#     for i in result:
#         if i!='。':
#             f.write(i)
#             f.write(' ')
#         else:
#             f.write('。')
#             f.write('\n')

# with open('juzi.txt','r+',encoding='utf-8') as f:
#     tmp=f.readlines()
# print(tmp)
# results=[]
# for i in tmp:
#     if i.strip()!='\n' and i!='\n' and i!=' \n':
#         results.append(i.strip())
#
# with open('juziresult.txt','w+',encoding='utf-8') as f:
#     for i in results:
#         f.write(i)
#         f.write('\n')

Time=[]
Rain=[]
Station=[]
Depart=[]
Measure=[]
Stain=[]
Area=[]
Point=[]
Water=[]
Level=[]

with open('合并.txt','r+',encoding='utf-8') as f:
    tmp=f.readlines()

print(tmp)
index=0
zt=False
tmp_Area=""
for i in tmp:
    try:
        if i!='':
            # print(i)
            if zt==False:
                if i.split(' ')[1]=='B-Area\n':
                    print('zhuadao')
                    zt=True
                    tmp_Area=""
                    tmp_Area=i.split(' ')[0]
                else:
                    pass
            else:
                if i.split(' ')[1] == 'I-Area\n':
                    tmp_Area=tmp_Area+i.split(' ')[0]
                else:
                    Area.append(tmp_Area)
                    zt=False
    except:
        print(i)
    index=index+1
print(Area)

out_list=["汉口地区","汉阳地区","武昌地区","武昌","洪山","经开","洪山区","经开区","硚口区",
          "汉阳区","高新区","长江","汉江","黄陂区","新洲区","江夏区","蔡甸区","新城区","汤逊湖","东湖"
          ,"北湖","南湖","江夏","中心城区","东西湖","其他地区","黄陂","武汉市","新洲",
          "东西湖","蔡甸",'开发区',"武汉",
 '汉阳',
 '江岸区',
 '江汉区',
 '汉口',
 '青山区',
 '武昌区',
 '武汉经开区',
 '东湖高新区',
 '东湖风景区',
 '硚口']


with open('area.txt','w',encoding='utf-8') as f:
    for i in Area:
        if i not in out_list:
            f.write(i)
            f.write('\n')

