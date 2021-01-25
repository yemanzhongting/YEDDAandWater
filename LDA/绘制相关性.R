library('corrplot')

library(corrgram)
install.packages("corrgram")
library(corrplot)
M <- cor(mtcars)
col1 <- colorRampPalette(c("#7F0000", "red", "#FF7F00", "yellow", "white",
                           "cyan", "#007FFF", "blue","#00007F"))
col2 <- colorRampPalette(c("#67001F", "#B2182B", "#D6604D", "#F4A582",
                           "#FDDBC7", "#FFFFFF", "#D1E5F0", "#92C5DE",
                           "#4393C3", "#2166AC", "#053061"))
col3 <- colorRampPalette(c("red", "white", "blue"))
col4 <- colorRampPalette(c("#7F0000", "red", "#FF7F00", "yellow", "#7FFF7F",
                           "cyan", "#007FFF", "blue", "#00007F"))
wb <- c("white", "black")
# methold参数设定不同展示方式
corrplot(M) #默认methold="circle"

# hclust.method = "ward.D2"设定聚类方法
corrplot(M, hclust.method = "ward.D2",type="lower")# addrect = 4),tl.pos="d"
corrplot(M, add=TRUE,order = "hclust",method = "number",type="upper",tl.pos="n",cl.pos="n")

M
corr1 <- read.csv("D:/QQ_receive/3152983713/FileRecv/corrdata.csv")
corr<-cor(corr1, method = "pearson")

#另外的方法
library(ggcor)
install.packages('ggcor')
library(rJava)
library(devtools)
library(usethis)
install.packages('rJava')
install.packages('devtools')
install.packages('usethis')

devtools::install_github("zlabx/ggcor")
devtools::install_github("briatte/ggcor")

devtools::install_github("yemanzhongting/ggcor")

install.packages('digest')

usethis::create_github_token()


usethis::edit_r_environ()

install.packages('corrplot')
library(corrplot)

corr <- fortify_cor(corr1, type = "upper", show.diag = TRUE,
                    cor.test = TRUE, cluster.type = "all")
mantel <- fortify_mantel(varespec, varechem,
                         spec.select = list(spec01 = 22:25,
                                            spec02 = 1:4,
                                            spec03 = 38:43,
                                            spec04 = 15:20),
                         mantel.fun = "mantel.randtest")
ggcor(corr, xlim = c(-5, 14.5)) + 
   add_link(mantel, diag.label = TRUE) +
   add_diaglab(angle = 45) +
   geom_square() + remove_axis("y")



corrplot(corr, hclust.method = "ward.D2",type="lower")# addrect = 4),tl.pos="d"
corrplot(corr,add=TRUE,order = "hclust",method = "number",type="upper",tl.pos="n",cl.pos="n")

#corrplot(b,add=TRUE, type="lower", method="number",diag=FALSE,tl.pos="n", cl.pos="n",col=col(10))

typeof(mtcars)
mtcars
library(readr)
new <- read_csv("C:/Users/Administrator/OneDrive - whu.edu.cn/自然灾害知识图谱/lianjia/new.csv")
View(new)
new
typeof(new)

C(new$year,new$Num)

N<- cor(newR)
corrplot(N) #默认methold="circle"

# hclust.method = "ward.D2"设定聚类方法
corrplot(N, hclust.method = "ward.D2",type="lower")# addrect = 4),tl.pos="d"
corrplot(N, add=TRUE,order = "hclust",method = "number",type="upper",tl.pos="n", cl.pos="n")
#corrplot(b,add=TRUE, type="lower", method="number",diag=FALSE,tl.pos="n", cl.pos="n",col=col(10))


res1 <- cor.mtest(newR, conf.level = 0.95) 
corrplot(N, method="ellipse",p.mat = res1$p, sig.level = 0.2,order = "AOE", type = "upper") #, tl.pos = "d"
corrplot(N, add = TRUE, p.mat = res1$p, sig.level = 0.2,type = "lower", method = "number", order = "AOE", 
         diag = FALSE, tl.pos = "n", cl.pos = "n") 

N<- cor(newR)
corrplot.mixed(N, lower = "number", upper = "circle", tl.col = "black",lower.col = "black", number.cex = 1)  
#tl.col 修改对角线的颜色,lower.col 修改下三角的颜色，number.cex修改下三角字体大小


library(readr)
fillnan <- read_csv("C:/Users/Administrator/Desktop/链家/fillnan.csv")
View(fillnan)
fillnan<-fillnan[,-1]
fillnan<-fillnan[,-2]
fillnan<-fillnan[,-2]
fillnan
corr1<- cor(fillnan)
corrplot.mixed(corr1, lower = "number", upper = "circle", tl.col = "black",lower.col = "red", number.cex = 1)  

library(readr)
delnan <- read_csv("C:/Users/Administrator/Desktop/链家/delnan.csv")
View(delnan)
delnan<-delnan[,-1]
delnan<-delnan[,-2]
delnan<-delnan[,-2]
delnan
corr2<- cor(delnan)
corrplot.mixed(corr2, lower = "number", upper = "circle", tl.col = "red",lower.col = "black", number.cex = 1)  

library(psych)

names(delnan)[1]<-"cases"
names(delnan)[2]<-"fee"
names(delnan)[6]<-"building"
names(delnan)[7]<-"house"
corr2<- cor(delnan)
corrplot.mixed(corr2, lower = "number", upper = "circle", tl.col = "red",lower.col = "black", number.cex = 1)

res1 <- cor.mtest(delnan, conf.level = .95)
corrplot(corr2,p.mat = res1$p, insig = "label_sig",
         sig.level = c( .01, .05), pch.cex = 1.5, pch.col = "black")

wilcox.test()

corr.test(delnan,use="complete")

corrplot.mixed(corr2, lower = "number", upper = "circle", tl.col = "red",lower.col = "black", number.cex = 1)

typeof(corr.test(delnan,use="complete"))
View(delnan)

library(stargazer)

library(car)
install.packages('car')

scatterplotMatrix

corr.p(delnan)

delnan

with(delnan,smoothScatter(cases,sky))

with(delnan,smoothScatter(cases,year))

library(car)

scatterplotMatrix(delnan)

stargazer(corr.test(delnan,use="complete")[3])
fillnan
fit<-lm(cases ~ fee+green+sky+year+building+house+price,data=delnan)

stargazer(summary(fit))


fivecol

colnames(fivecol) <- c( 'D0','D1','D2','D3','D4','D5','D6','D7','D8','D9') 

matrix <- cor (fivecol)

#直接画图，不设置其它参数
corrplot(corr=matrix)


corrplot(matrix, method = "square")

corrplot(matrix,method = "pie")

getSig <- function(dc) {
   sc <- ''
   if (dc < 0.01) sc <- '***'
   else if (dc < 0.05) sc <- '**'
   else if (dc < 0.1) sc <- '*'
   sc
}

res1 <- cor.mtest(matrix)

res1$p

corrplot(matrix, p.mat = res1$p, tl.pos = "d",type ="lower",insig = "label_sig",
         sig.level = c(.005, .01, .05), pch.cex = .9,
         pch.col = "black")



corrplot(matrix, add = TRUE, type = "upper", method = "circle", diag = FALSE, tl.pos = "n", cl.pos = "n")


`sim` <- read.csv("E:/Githubresponsity/YEDDA/LDA/10X10sim.csv", header=TRUE)

matrix

simdata=as.matrix(sim)

corrplot(simdata, add = TRUE, type = "upper", method = "number"
         , diag = FALSE
         , tl.pos = "n", cl.pos = "n")

