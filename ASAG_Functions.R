library(tm)          
library(text2vec)
library(stringr)
library(dplyr)
library(ggplot2)
library(caTools )


compute_feature = function(ques, ans, key){
  preprocess <- function(vect){
    vect = tolower(vect)
    vect = removePunctuation(vect)
    vect = removeWords(vect, stopwords("en"))
    vect
  }
  
  ans = preprocess(ans)
  key = preprocess(key)
  ques = preprocess(ques)
  ans = stemDocument(ans)
  key = stemDocument(key)
  ques = stemDocument(ques)
  
  it <- itoken(ans, progressbar = FALSE)
  vocab <- create_vocabulary(it) #term; term count; doc count
  vectorizer <- vocab_vectorizer(vocab) 
  dtm <- create_dtm(it, vectorizer)
  tfidf <- TfIdf$new()
  dtm_tfidf <- fit_transform(dtm, tfidf)
  
  mat_tfidf = as.matrix(dtm_tfidf)
  
  rvector <- colnames(mat_tfidf)
  
  score_vect <- rep(0., times = length(ans))
  len_vect <- rep(0., times = length(ans))
  
  count_keywords <- function(document, keywords) {
    sapply(keywords, function(word) sum(str_count(document, fixed(word))))
  }
  
  term_frequencies <- count_keywords(ans, rvector)
  l2 <- rep(0, length(rvector))
  l2 = term_frequencies[colnames(mat_tfidf)]/20 + l2
  question_words <- strsplit(ques, "\\s+")[[1]]
  words_to_halve <- intersect(names(l2), question_words)
  l2[words_to_halve] <- l2[words_to_halve] / 1.8
  
  
  
  for (i in c(1:length(ans))){
    X <- ans[i] 
    Y <- key 
    
    X_set <- unlist(strsplit(X, "\\s+"))
    Y_set <- unlist(strsplit(Y, "\\s+"))
    len_vect[i] = length(X_set) / length(Y_set)
    
    X_set <- unlist(X_set) 
    Y_set <- unlist(Y_set)  
    
    l1 <- rep(0, length(rvector))  
    #l2 <- rep(0, length(rvector))  
    
    # Update the binary vectors based on the word overlap
    for (j in 1:length(rvector)) {
      if (rvector[j] %in% X_set) l1[j] <- 1  
      if (rvector[j] %in% Y_set) l2[j] <- 1  
    }
    
    # Add the TF-IDF values
    l1 = mat_tfidf[i,] + l1
    #l2 = term_frequencies[colnames(mat_tfidf)]/20 + l2
    
    # Compute cosine similarity
    c <- sum(l1 * l2)  # Dot product
    cosine_similarity <- c / sqrt(sum(l1) * sum(l2))  # Normalize
    cosine_similarity
    cosine_similarity = ifelse(is.nan(cosine_similarity), 0, cosine_similarity)
    score_vect[i] = cosine_similarity
  }
  score_vect = score_vect * 5
  retval = list(score = score_vect, length = len_vect)
  
  
  retval
}

comp_all = function(dat){
  nq = length(unique(dat$question))
  q_v = unique(dat$question)
  for(i in 1:nq){
    qi = q_v[[i]]
    ai = dat$student_answer[dat$question == qi]
    ki = dat$desired_answer[dat$question == qi][[1]]
    feat_vect = compute_feature(qi, ai, ki)
    dat$feature[dat$question == qi] = feat_vect$score
    dat$len_ratio[dat$question == qi] = feat_vect$length
  }
  dat
}

pred = function(dat,mod){
  dat = comp_all(dat)
  score = predict(mod,dat["feature"])
  dat$score = score
  dat
}



validate = function(tr,ts){
  # Plot the polynomial regression curve
  a = ggplot(tr, aes(x = feature, y = score_avg)) +
    geom_point(color = "blue", alpha = 0.5) +  # Scatter plot of training data
    geom_smooth(aes(y = score_avg), method = "lm", formula = y ~ poly(x, 7, raw = TRUE), color = "red") +  # Regression curve
    geom_point(data = ts, aes(x = feature, y = predicted), color = "green", shape = 17, size = 2) +  # ts predictions
    labs(title = "Polynomial Regression Curve", x = "Feature", y = "Score Average")
  
  c = ggplot(ts, aes(x=err)) +
    geom_density(alpha = 0.2, fill = "blue") +
    labs(title = "Distribution of Error", x = "error")
  
  d = ggplot(ts, aes(x=err^2)) +
    geom_density(alpha = 0.2, fill = "blue") +
    labs(title = "Distribution of Error Squared", x = "error^2")
  
  mean_abs_err = mean(abs(ts$err), na.rm = TRUE)
  mean_sq_err = mean(ts$err^2, na.rm = TRUE)
  rmse = mean_sq_err^0.5
  
  cat("Mean Absolute error : ", mean_abs_err, "\nMean Square error : ", mean_sq_err, "\nRoot Mean Square error : ", rmse)
  print(a)
  #print(b)
  print(c)
  print(d)
}


model2 = function(subs){
  key = subs$desired_answer[[1]]
  ans = subs[["student_answer"]]
  tempvar = rep(0,length(ans))
  
  for (i in c(1:length(ans))){
    X<-ans[i]
    Y<-key
    
    X_list <- unlist(strsplit(tolower(X), "\\s+"))
    Y_list <- unlist(strsplit(tolower(Y), "\\s+"))
    
    sw <- stopwords("en")
    
    X_set <- setdiff(X_list, sw)
    Y_set <- setdiff(Y_list, sw)
    rvector <- union(X_set, Y_set)
    
    l1 <- rep(0, length(rvector))
    l2 <- rep(0, length(rvector))
    
    for (i in 1:length(rvector)) {
      if (rvector[i] %in% X_set) l1[i] <- 1
      if (rvector[i] %in% Y_set) l2[i] <- 1
    }
    
    c <- sum(l1 * l2)
    cosi <- c / sqrt(sum(l1) * sum(l2))
    tempvar[i] <- cosi
  }
  tempvar
}