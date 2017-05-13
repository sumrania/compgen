# BoxPLot using ggplot2


# ##set current working directory to folder titled 'Summary_plots"
# setwd("Desktop/comp_genomics_project/")

library(ggplot2)
library(reshape2)
library(superheat)
library(gplots)
library(RColorBrewer)


early_g1 <- read.table("early_G1.txt",sep="\t",header=F)
late_g1 <- read.table("late_G1.txt",sep="\t",header=F)
s <- read.table("S.txt",sep="\t",header=F)
g2 <- read.table("G2.txt",sep="\t",header=F)
m <- read.table("M.txt",sep="\t",header=F)

centre_return <- function(data){
  return (data - apply(data,2,mean))
}


melt_bind <-function(data,phase_label){
  time_pts <- rep(1:17,each=length(data[,1]))
  melted_data <- melt(data)
  label <- rep(phase_label)
  return (cbind(melted_data,label,time_pts))
  
}

preprocess_melt <- function(data,label){
  data_centered <- centre_return(data)
  data_melted <- melt_bind(data_centered,label)
  return (data_melted)  
}


m_for_plot <- preprocess_melt(m,"m")
g2_for_plot <- preprocess_melt(g2,"g2")
s_for_plot <- preprocess_melt(s,"s")
early_g1_for_plot <- preprocess_melt(early_g1,"early_g1")
late_g1_for_plot <- preprocess_melt(late_g1,"late_g1")

all_time_series <- rbind(early_g1_for_plot,late_g1_for_plot,s_for_plot,g2_for_plot,m_for_plot)

plt.boxes.time.series <- function(data,plot_label){
  # data_centred <- data -apply(data,2,mean)
  time_pts <- rep(1:17,each=length(data[,1]))
  data.melt <- melt(data)
  plot_data <- cbind(data.melt,time_pts)
  plt_temp <- ggplot(plot_data,aes(x=time_pts,y=value,group=time_pts)) + geom_boxplot(fill="green",colour="black",outlier.colour = "purple",outlier.shape = 5)
  plt_temp <- plt_temp + labs(title=plot_label,x="Time Points",y="Log Normalized Expression")
  return (plt_temp)
}

# individual box plots for each phase
plt.boxes.time.series(m,"m phase")

#facet wrap with dark hue
plt <- ggplot(all_time_series,aes(x=time_pts,y=value,fill=label,group=time_pts)) + geom_boxplot(outlier.color = "darkred",outlier.shape = 5) + facet_wrap(~label) + scale_fill_hue(l=40, c=35) + labs(title="Boxplots of Expression Trends",y="Normalized Expression values",x="Time Points") + theme(plot.title = element_text(hjust = 0.5))
plt
