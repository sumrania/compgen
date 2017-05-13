# heatmap using superheat

# ##set current working directory to folder titled 'Summary_plots"
# setwd("Desktop/comp_genomics_project/")

library(ggplot2)
library(reshape2)
library(superheat)
library(gplots)
library(RColorBrewer)

cho_dataset <- read.table("heat_map_cho_385.txt",header=FALSE,sep="\t")
data_heatmap_cho <- cho_dataset[,2:18]
data_heatmap_cho[is.na(data_heatmap_cho)] <- 0
colnames(data_heatmap_cho) <- c("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17")

memberships <- rep("1",66)
memberships <- c(memberships,rep("2",132))
memberships <- c(memberships,rep("3",74))
memberships <- c(memberships,rep("4",50))
memberships <- c(memberships,rep("5",55))


### Smoothed heatmap
superheat(data_heatmap_cho,heat.pal = c("red", "black", "green"),membership.rows = memberships,smooth.heat = TRUE,
          title="HEATMAP: Tandem Cell Cycles",title.size = 10,
          row.title = "Phase",column.title = "Time Points",row.title.size = 6,column.title.size = 6)

### Heatmap with all ~380 genes! complete resolution
superheat(data_heatmap_cho,heat.pal = c("red", "black", "green"),membership.rows = memberships,smooth.heat = FALSE,
          title="HEATMAP: Tandem Cell Cycles",title.size = 10,
          row.title = "Phase",column.title = "Time Points",row.title.size = 6,column.title.size = 6)

