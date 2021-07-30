library(ggplot2)
library(Rmisc)

dat <- read.csv("../../evaluations/evaluation-simulated-2021-07-30_20.29.13.csv")

summarized = summarySE(dat, measurevar="accuracy", groupvars=c("iteration", "gar"))
pd <- position_dodge(.1)

p <- ggplot(summarized, aes(x=iteration, y=accuracy, group=gar, colour=gar)) +
     geom_line() +
     geom_errorbar(aes(ymin=accuracy-se, ymax=accuracy+se), width=.1, position=pd) +
     ylim(c(0, 1)) +
     theme_bw() +
     xlab("Iteration") +
     ylab("Accuracy")

ggsave(filename="accuracy.pdf", plot=p, height=4, width=8)
