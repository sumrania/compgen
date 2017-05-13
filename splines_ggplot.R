# smooth spline

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

# #plot individual phase/class time series 
plt.points.time.series <- function(data,plot_label){
  data_centred <- data - apply(data,2,mean)
  time_pts <- rep(1:17,each=length(data[,1]))
  data.melt <- melt(data_centred)
  plot_data <- cbind(data.melt,time_pts)
    plt1 <- ggplot(plot_data) + geom_point(aes(x=time_pts,y=value,group=time_pts),color="green",alpha=0.2) + geom_smooth(aes(x=time_pts,y=value),method='lm',formula=y~splines::bs(x,3),se=FALSE,color="purple") + ylim(-1.0,1.0)
  plt1 <- plt1 + labs(title=plot_label,x="Time Points",y="Log Normalized Expression")  
  plt1 <- plt1 + theme(plot.title=element_text(size=30, face="bold"), 
                       axis.text.x=element_text(size=10), 
                       axis.text.y=element_text(size=10),
                       axis.title.x=element_text(size=20),
                       axis.title.y=element_text(size=20))
  plt1 <- plt1 + theme(legend.position="none",plot.title = element_text(hjust=0.5))
  plt1 <- plt1 + geom_vline(xintercept = 9,color="black",size=0.8,linetype=3)
  return (plt1)
}

#plot individual time series with cubic splines ##change dataset and title as arguments
plt.points.time.series(early_g1,"early g1")


# plot all classes/phases without points, only splines

plt.trends<- function(data,plot_label){
  ## cubic splines
  # plt_temp <- ggplot(all_time_series) + geom_smooth(aes(x=time_pts,y=value,color=label,group=label),
                    # se=FALSE,method="lm",formula=y~splines::bs(x,3)) + geom_vline(xintercept = 9,color="black",size=0.8,linetype=3)
  ## gam method
  plt_temp <- ggplot(all_time_series) + geom_smooth(aes(x=time_pts,y=value,color=label,group=label),se=FALSE) + geom_vline(xintercept = 9,color="black",size=0.8,linetype=3)
  plt_temp <- plt_temp +  labs(title="TEMPORAL PATTERNS",x="Time Points",y="Log Normalized Expression",color="Cell Cycle Phase")
  plt_temp <- plt_temp + theme(plot.title = element_text(lineheight = 3,face="bold",size=24,hjust=0.5),
                               axis.text=element_text(size=10),axis.title=element_text(size=14))
  return (plt_temp)
}

plt <- plt.trends(all_time_series,"Temporal Patterns")
plt
