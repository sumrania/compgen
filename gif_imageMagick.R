# ##create gif file using ImageMagick
# requires local installation of ImageMagick


# Set directory
setwd("Desktop/comp_genomics_project/")

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

phase_data <- list(early_g1_for_plot,late_g1_for_plot,s_for_plot,g2_for_plot,m_for_plot)

plt.mov <- function(data){
  for(i in 1:5){
    png(file=paste("phase",i,".png"), width=500, height=500)
    plt_temp <- ggplot(data[[i]]) + geom_smooth(aes(x=time_pts,y=value),method="loess",se=FALSE) 
    print(plt_temp)
    # Sys.sleep(1)
    dev.off()
  }
  system("convert -delay 50 *.png animation_phases.gif")        
}
# create gif file of phase transitions
plt.mov(phase_data)
