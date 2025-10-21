library(tm)          
library(text2vec)
library(stringr)
library(dplyr)
library(ggplot2)
library(caTools)
library(dplyr)


df = read.csv("mohler_dataset.csv")
df <- df %>% filter(student_answer != "#NAME?")

q_vect = unique(df$question)
last_seed = .Random.seed
.Random.seed = readRDS(file = "seed.rds")
train_test_filter = sample.split(q_vect, SplitRatio = 0.8)
training_q = subset(q_vect, train_test_filter == TRUE)
testing_q = subset(q_vect, train_test_filter == FALSE)

train_df = df[df$question %in% training_q, ]
testing_df = df[df$question %in% testing_q, ]
test_inp = testing_df[c("question", "student_answer", "desired_answer")]
test_score = testing_df["score_avg"]

train_df = comp_all(train_df)

write.csv(train_df,"training_data.csv")
write.csv(test_inp,"testing_data.csv")

model_inp = train_df[c("feature","score_avg")]

synth_0 = data.frame(score_avg = rep(0,100), feature = rep(0,100))
model_df = rbind(model_inp, synth_0)

df_smote = data.frame(
   feature = rnorm(100, mean = 0, sd = 1),
   score_avg = rnorm(100, mean = 0, sd = 1)
)
df_smote = df_smote[df_smote$feature>0 & df_smote$score_avg>0,]
model_inp = rbind(model_inp, df_smote)

model_p = lm(score_avg ~ poly(feature ,7 , raw = TRUE), data = model_inp)

test_inp = comp_all(test_inp)
res = pred(test_inp,model_p)
score = res$score

score[score>5] = 5
score[score<0] = 0

results <- data.frame(score_avg = test_score$score_avg, predicted = score, feature = res$feature)
results$err = results$predicted - results$score_avg

validate(model_inp,results)


df_rmse <- results %>%
  group_by(score_avg) %>%
  summarize(rmse = sqrt(mean(err^2, na.rm = TRUE))) %>%
  ungroup()

# 2. Plot RMSE vs feature
# Since feature is discrete, a line plot or point plot is more appropriate
ggplot(df_rmse, aes(x = score_avg, y = rmse)) +
  geom_line() +
  geom_point() +
  labs(
    title = "RMSE by score",
    x = "score_avg",
    y = "RMSE"
  ) +
  theme_minimal()


ggplot(df_rmse, aes(x = score_avg, y = rmse)) +
  geom_point() +
  geom_smooth(se = TRUE, method = "loess", fill = "gray", color = "red") +
  labs(
    title = "Smoothed RMSE Trend by Score",
    x = "Score",
    y = "RMSE"
  ) +
  theme_minimal()
