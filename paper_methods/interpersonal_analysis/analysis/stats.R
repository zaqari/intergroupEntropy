library(stringdist)
library(lme4)
library(gridExtra)
library(ggplot2)

#
#
# LOAD, TIDY AND CODE PRE-PROCESSED RAW DATA 
#
#

a = read.csv('../results.csv',header=T)

a$abs_dist = abs(a$i-a$j) # distances (k)
a$rel_dist = a$i-a$j

a = a[a$n_i>5&a$n_j>5,] # tokens must be 5 and up in convo 
a = a[a$abs_dist>0 & a$abs_dist<=10,] # k <= 10

dim(a)

a$lev = stringsim(a$txt_i,a$txt_j)

a$bl = FALSE
a$bl[a$baseline!=''] = TRUE # baseline factor

a$self = a$who_i==a$who_j 

table(a$bl) # number of turns per baseline/experimental

a[1:2,] # glance

# factor by length of relevant utterance
a$h_1_n = as.numeric(a$h_1)/as.numeric(a$n_i)
a$h_2_n = as.numeric(a$h_2)/as.numeric(a$n_j)

#
#
# LINEAR MODELS TESTING VARIOUS ASSOCIATIONS WITH ENTROPY
#
#

# raw, divided by n in relevant comparison
modl = lmer(h_1_n~self*bl*abs_dist+(1|fl)+(1|who_i)+(1|who_j),data=a)
coefs = data.frame(coef(summary(modl)))
coefs$p = round(1-pnorm(abs(coefs$t.value)),5)
coefs

# string similarity check... residualize
a$residH = resid(lm(h_1_n~lev,data=a))
modl = lmer(residH~self*bl*abs_dist+(1|fl)+(1|who_i)+(1|who_j),data=a)
coefs = data.frame(coef(summary(modl)))
coefs$p = round(1-pnorm(abs(coefs$t.value)),5)
coefs

#
#
# PLOTS OF ENTROPY ACROSS k SHOWING BASELINE, OBSERVED; SELF, OTHER
#
#

se = function(x) sd(x) / sqrt(length(x))

b_mean = aggregate(residH~rel_dist+bl, data=a, mean)
b_se = aggregate(residH~rel_dist+bl, data=a, se)
b1 = merge(b_mean, b_se, by=c("rel_dist", "bl"))
colnames(b1) = c("rel_dist", "bl", "residH_mean", "residH_se")

# Residualized H x relative distance (k)
p1 = ggplot(b1, aes(x=rel_dist, y=residH_mean)) +
  geom_ribbon(aes(ymin=residH_mean-residH_se, ymax=residH_mean+residH_se, fill=as.factor(!bl)), alpha=0.3) +
  geom_line(aes(color=as.factor(!bl))) +
  labs(x="Relative distance (k)", y="Residualized entropy") +
  scale_color_manual(values=c("red", "blue"), labels=c("Baseline", "Observed")) +
  scale_fill_manual(values=c("red", "blue"), labels=c("Baseline", "Observed")) +
  theme(legend.position="bottomright")
  # *** NB: flipped "bl" with !bl in color spec to make color consistent with next panel: ***

# Self/other/baseline for |k|
b2_mean = aggregate(residH~abs_dist+self, data=a[!a$bl,], mean)
b2_se = aggregate(residH~abs_dist+self, data=a[!a$bl,], se)
b3_mean = aggregate(residH~abs_dist, data=a[a$bl,], mean)
b3_se = aggregate(residH~abs_dist, data=a[a$bl,], se)

b2 = merge(b2_mean, b2_se, by=c("abs_dist", "self"))
b3 = merge(b3_mean, b3_se, by="abs_dist")
colnames(b2) = c("abs_dist", "self", "residH_mean", "residH_se")
colnames(b3) = c("abs_dist", "residH_mean", "residH_se")

df_combined = rbind(b2, transform(b3, self='baseline'))

p2 = ggplot(df_combined, aes(x=abs_dist, y=residH_mean, color=as.factor(self))) +
  geom_ribbon(aes(ymin=residH_mean-residH_se, ymax=residH_mean+residH_se, fill=as.factor(self)), alpha=0.3) +
  geom_line() +
  labs(x="Absolute distance (|k|)", y="Residualized entropy") +
  scale_color_manual(values=c("red", "green", "blue"), labels=c("Baseline", "Other", "Self")) +
  scale_fill_manual(values=c("red", "green", "blue"), labels=c("Baseline", "Other", "Self")) +
  theme(legend.position="bottomright")

grid.arrange(p1, p2, ncol=2)



# to check / confirm the ggplot output

# this is unresidualized; not shown but results consistent; checking
b = aggregate(h_1_n~rel_dist+bl,data=a,mean);ylab='Unresidualized entropy'
plot(b[!b$bl,c(1,3)],type='o',xlab='Relative distance (k)',ylab=ylab,ylim=c(.25,.3),lwd=2) # observed
points(b[b$bl,c(1,3)],type='o',col='red',lwd=2) # baseline

par(mfrow=c(1,2))

b = aggregate(residH~rel_dist+bl,data=a,mean);ylab='Residualized entropy'
plot(b[!b$bl,c(1,3)],type='o',xlab='Relative distance (k)',ylab=ylab,ylim=c(-0.02,.01),lwd=2) # observed
points(b[b$bl,c(1,3)],type='o',col='red',lwd=2) # baseline
legend('bottomright',lwd=2,col=c('black','red'),c('observed','baseline'),cex=.7,bg='transparent',bty='n')

b = aggregate(residH~abs_dist+self,data=a[!a$bl,],mean)
plot(b[b$self,]$abs_dist,b[b$self,]$residH,type='o',ylim=c(-.02,.01),
     xlab='Absolute distance (|k|)',ylab='Residualized entropy',lwd=2)
points(b[!b$self,]$abs_dist,b[!b$self,]$residH,type='o',col='green',lwd=2)
b = aggregate(residH~abs_dist,data=a[a$bl,],mean,lwd=2)
points(b$abs_dist,b$residH,type='o',ylim=c(-.1,.1),col='red',lwd=2)
legend('bottomright',lwd=2,col=c('black','green','red'),c('self','other','baseline'),cex=.7,bg='transparent',bty='n')
