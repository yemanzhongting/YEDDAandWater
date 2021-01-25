library(tidyverse)
install.packages('ragg')
library(ragg)



# 创建一个数据 ------------------------------------------------------------------


data <- data.frame(
  individual=paste( "Mister ", seq(1,60), sep=""),
  group=c( rep('A', 10), rep('B', 30), rep('C', 14), rep('D', 6)) ,
  value=sample( seq(10,100), 60, replace=TRUE)
)


# 接下来就是处理数据，对数据做转换 --------------------------------------------------------


datagroup <- data$group %>% unique()

allplotdata <- tibble('group' = datagroup,
                      'individual' = paste0('empty_individual_', seq_along(datagroup)),
                      'value' = 0) %>% 
  bind_rows(data) %>% arrange(group) %>% mutate(xid = 1:n()) %>% 
  mutate(angle = 90 - 360 * (xid - 0.5) / n()) %>% 
  mutate(hjust = ifelse(angle < -90, 1, 0)) %>% 
  mutate(angle = ifelse(angle < -90, angle+180, angle)) 


# 这个是提取出空的数据，做一些调整 --------------------------------------------------------


firstxid <- which(str_detect(allplotdata$individual, pattern = "empty_individual"))

segment_data <- data.frame('from' = firstxid + 1,
                           'to' = c(c(firstxid - 1)[-1], nrow(allplotdata)),
                           'label' = datagroup) %>% 
  mutate(labelx = as.integer((from + to)/2))

# 这个是自定坐标轴 ----------------------------------------------------------------


coordy <- tibble('coordylocation' = seq(from = min(allplotdata$value), to = max(allplotdata$value), 10),
                 'coordytext' = as.character(round(coordylocation, 2)),
                 'x' = 1)

# 这个是自定义坐标轴的网格 ------------------------------------------------------------


griddata <- expand.grid('locationx' = firstxid[-1], 'locationy' = coordy$coordylocation)


# 这个就是开始画图了 ---------------------------------------------------------------


p <- ggplot() + 
  geom_bar(data = allplotdata, aes(x = xid, y = value, fill = group), stat = 'identity') + 
  geom_text(data = allplotdata %>% filter(!str_detect(individual, pattern = "empty_individual")), 
            aes(x = xid, label = individual, y = value+10, angle = angle, hjust = hjust),
            color="black", fontface="bold",alpha=0.6, size=2.5) + 
  geom_segment(data = segment_data, aes(x = from, xend = to), y = -5, yend=-5) + 
  geom_text(data = segment_data, aes(x = labelx, label = label), y = -15) + 
  geom_text(data = coordy, aes(x = x, y = coordylocation, label = coordytext),
            color="grey", size=3 , angle=0, fontface="bold") + 
  geom_segment(data = griddata, 
               aes(x = locationx-0.5, xend = locationx + 0.5, y = locationy, yend = locationy),
               colour = "grey", alpha=0.8, size=0.6) + 
  scale_x_continuous(expand = c(0, 0)) + 
  scale_y_continuous(limits = c(-50,100)) + 
  coord_polar() +
  theme_void() +
  labs(title = "pypi") + 
  theme(legend.position = 'none')

p


# 这里就是保存了 -----------------------------------------------------------------


ggsave(filename = 'circularbar2.png', plot = p, width = 10, height = 10, device = ragg::agg_png())