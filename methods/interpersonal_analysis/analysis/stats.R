library(stringdist)
library(lme4)

# load, check
a = read.csv('../test.csv',header=T)
a$abs_dist = abs(a$i-a$j)
a$rel_dist = a$i-a$j
a = a[a$n_i>5&a$n_j>5,]
a = a[a$abs_dist>0 & a$abs_dist<=10,]
dim(a)

a$lev = stringdist::stringsim(a$txt_i,a$txt_j)

a$bl = FALSE
a$bl[a$baseline!=''] = TRUE

a$self = a$who_i==a$who_j

table(a$bl)

a[1:2,]

# factor by length of relevant utterance
a$h_1_n = as.numeric(a$h_1)/as.numeric(a$n_i)
a$h_2_n = as.numeric(a$h_2)/as.numeric(a$n_j)

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

b = aggregate(residH~rel_dist,data=a,mean)
b = aggregate(h_1_n~rel_dist,data=a,mean)
b = aggregate(h_1_n~abs_dist,data=a,mean)
b = aggregate(lev~rel_dist,data=a,mean)
plot(b,type='o')

b = aggregate(residH~abs_dist+self,data=a[!a$bl,],mean)
plot(b[b$self,]$abs_dist,b[b$self,]$residH,type='o',ylim=c(-.1,.1))
points(b[!b$self,]$abs_dist,b[!b$self,]$residH,type='o',col='green')
b = aggregate(residH~abs_dist,data=a[a$bl,],mean)
points(b$abs_dist,b$residH,type='o',ylim=c(-.1,.1),col='red')

modl = lmer(lev~self*tb*abs_dist+(1|fl)+(1|who_i)+(1|who_j),data=a)
coefs = data.frame(coef(summary(modl)))
coefs$p = round(1-pnorm(abs(coefs$t.value)),5)
coefs


