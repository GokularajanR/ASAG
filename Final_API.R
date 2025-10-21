library(glue)
library(plumber)
library(htmltools)
library(ggplot2)
library(tidyr)
library(dplyr)
library(base64enc)
library(tidytext)
library(stringr)

# --- N-gram Overlap Method (Model R) ---

get_ngrams <- function(text, n_gram_size = 2) {
  if (is.na(text) || text == "") {
    return(character(0)) # Return empty vector for empty/NA input
  }
  
  text_df <- tibble(text = text)
  
  ngrams <- text_df %>%
    unnest_tokens(ngram, text, token = "ngrams", n = n_gram_size) %>%
    distinct(ngram) %>% # Get unique n-grams
    pull(ngram)
  
  return(ngrams)
}


pred_r <- function(test_input, n_gram_size = 2) {
  if (nrow(test_input) == 0) {
    return(data.frame(name=character(), score=numeric()))
  }
  
  required_cols <- c("name", "student_answer", "desired_answer")
  if (!all(required_cols %in% names(test_input))) {
    stop(paste("Input data frame must contain columns:", paste(required_cols, collapse=", ")))
  }
  
  scores <- sapply(seq_len(nrow(test_input)), function(i) {
    student_ans <- test_input$student_answer[i]
    desired_ans <- test_input$desired_answer[i]
    
    student_ngrams <- get_ngrams(student_ans, n_gram_size = n_gram_size)
    desired_ngrams <- get_ngrams(desired_ans, n_gram_size = n_gram_size)
    
    intersection_ngrams <- intersect(student_ngrams, desired_ngrams)
    union_ngrams <- union(student_ngrams, desired_ngrams)
    
    if (length(union_ngrams) == 0) {
      if (length(desired_ngrams) == 0 && length(student_ngrams) == 0) {
        jaccard_index <- 1.0 # Both empty, perfect match of nothing
      } else {
        jaccard_index <- 0.0 
      }
    } else {
      jaccard_index <- length(intersection_ngrams) / length(union_ngrams)
    }
    
    final_score <- round(jaccard_index * 100)
    
    final_score <- max(0, min(100, final_score))
    
    return(final_score)
  })
  
  output_df <- data.frame(
    name = test_input$name,
    score = scores
  )
  
  return(output_df)
}

# --- Content Word Extraction Method (Model Q) ---

get_content_words <- function(text) {
  if (is.na(text) || text == "") {
    return(character(0)) # Return empty vector for empty/NA input
  }
  
  text_df <- tibble(text = text)
  
  content_words <- text_df %>%
    unnest_tokens(word, text, token = "words") %>%
    anti_join(stop_words, by = "word") %>%
    filter(!str_detect(word, "^[0-9.]+$")) %>%
    # mutate(word = SnowballC::wordStem(word)) %>% # Optional stemming
    distinct(word) %>%
    pull(word)
  
  return(content_words)
}

# This corresponds to Model Q
pred_q <- function(test_input) {
  # 1. Handle Empty Input
  if (nrow(test_input) == 0) {
    return(data.frame(name=character(), score=numeric()))
  }
  
  # Check for required columns (optional but good practice)
  required_cols <- c("name", "student_answer", "desired_answer")
  if (!all(required_cols %in% names(test_input))) {
    stop(paste("Input data frame must contain columns:", paste(required_cols, collapse=", ")))
  }
  
  # 2. Process row by row
  scores <- sapply(seq_len(nrow(test_input)), function(i) {
    # Get student answer and desired answer for the current row
    student_ans <- test_input$student_answer[i]
    desired_ans <- test_input$desired_answer[i]
    
    # Extract content words (keywords) from the desired answer
    keywords <- get_content_words(desired_ans)
    
    # Handle case where the desired answer has no content words
    if (length(keywords) == 0) {
      student_words_check <- get_content_words(student_ans)
      if(length(student_words_check) == 0) return(100) else return(0)
    }
    
    # Extract content words from the student answer
    student_words <- get_content_words(student_ans)
    
    # Calculate overlap: number of keywords found in student answer
    keywords_found <- sum(keywords %in% student_words) # Count matches
    
    # Calculate score as a proportion (0 to 1)
    score_proportion <- keywords_found / length(keywords)
    
    # Scale score to 0-100 and round
    final_score <- round(score_proportion * 100)
    
    # Ensure score is strictly between 0 and 100 (optional, but good practice)
    final_score <- max(0, min(100, final_score))
    
    return(final_score)
  })
  
  # 3. Format Output
  output_df <- data.frame(
    name = test_input$name,
    score = scores
  )
  
  return(output_df)
}

# --- Vector Space Model (TF-IDF Cosine Similarity) (Model P) ---
# Using the dummy 'pred' function as placeholder, assuming it represents Model P
# In reality, this would be a more complex TF-IDF + Cosine Similarity function
# like the one discussed earlier.

pred <- function(test_input, model) {
  # This is the DUMMY placeholder for Model P (Vector Space)
  # Replace this with your actual TF-IDF implementation if available
  if (nrow(test_input) == 0) {
    return(data.frame(name=character(), score=numeric()))
  }
  scores <- sapply(seq_len(nrow(test_input)), function(i) {
    # Simple demo: score based on length difference (closer is better)
    len_diff <- abs(nchar(test_input$student_answer[i]) - nchar(test_input$desired_answer[i]))
    max_len <- max(nchar(test_input$student_answer[i]), nchar(test_input$desired_answer[i]), 1) # Avoid div by zero
    score <- max(0, 1 - (len_diff / max_len))
    round(score * 100) # Score out of 100
  })
  return(data.frame(name = test_input$name, score = scores))
}

# Dummy model placeholder (just needed for the function call signature)
model_p <- "vector_space_model"


# --- Plotting and Helper Functions ---

# --- Function to Map Internal Variable Names to Display Names ---
map_score_var_to_name <- function(var_name) {
  case_when(
    var_name == "score_p" ~ "Vector Space Score",
    var_name == "score_q" ~ "Keyword Extraction Score",
    var_name == "score_r" ~ "N-gram Overlap Score",
    TRUE ~ var_name # Fallback
  )
}

map_model_code_to_name <- function(model_code) {
  case_when(
    model_code == "P" ~ "Vector Space Model",
    model_code == "Q" ~ "Keyword Extraction Model",
    model_code == "R" ~ "N-gram Overlap Model",
    TRUE ~ model_code # Fallback
  )
}

# --- Function to Generate Score Heatmap ---
generate_score_heatmap <- function(scores_long_df) {
  # Ensure required columns exist and data is present
  if (!all(c("name", "model", "score") %in% names(scores_long_df)) || nrow(scores_long_df) == 0) {
    warning("Missing columns or no data for heatmap.")
    return(NULL)
  }
  
  # Ensure model names are factors with desired order for Y axis
  # Assuming the mapping to full names already happened before calling this function
  desired_levels <- c("Vector Space Model", "Keyword Extraction Model", "N-gram Overlap Model")
  scores_long_df$model <- factor(scores_long_df$model, levels = intersect(desired_levels, unique(scores_long_df$model)))
  
  # Convert names to factors for ordering on X axis (optional, default is alphabetical)
  scores_long_df$name <- factor(scores_long_df$name, levels = sort(unique(scores_long_df$name)))
  
  p <- ggplot(scores_long_df, aes(x = name, y = model, fill = score)) +
    geom_tile(color = "white", size = 0.5) + # Add white lines between tiles
    geom_text(aes(label = round(score)), size = 3, color="black") + # Add score text in center
    scale_fill_gradient(low = "lightblue", high = "darkblue", limits=c(0, 100), na.value = "grey80") +
    labs(
      title = "Score Heatmap: Students vs. Grading Models", # Updated Title
      x = "Student Name",
      y = "Grading Model", # Updated Axis Label
      fill = "Score"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1), # Rotate x labels if many students
      panel.grid.major = element_blank(), # Remove grid lines
      panel.grid.minor = element_blank()
    )
  
  # Save, encode, and return base64 URI
  temp_plot_file <- tempfile(fileext = ".png")
  ggsave(temp_plot_file, plot = p, width = max(6, length(unique(scores_long_df$name))*0.5), height = max(4, length(unique(scores_long_df$model))*0.5), units = "in", dpi = 96) # Adjusted size calculation
  
  if (file.exists(temp_plot_file)) {
    img_data <- readBin(temp_plot_file, "raw", file.info(temp_plot_file)$size)
    base64_img <- base64enc::dataURI(data = img_data, mime = "image/png")
    unlink(temp_plot_file)
    return(base64_img)
  } else {
    warning("Failed to save the heatmap image.")
    return(NULL)
  }
}

# --- Function to Generate Stacked Bar Plot ---
generate_stacked_bar_plot <- function(scores_long_df) {
  # Ensure required columns exist and data is present
  if (!all(c("name", "model", "score") %in% names(scores_long_df)) || nrow(scores_long_df) == 0) {
    warning("Missing columns or no data for stacked bar plot.")
    return(NULL)
  }
  
  # Order models consistently using full names
  desired_levels <- c("Vector Space Model", "Keyword Extraction Model", "N-gram Overlap Model")
  scores_long_df$model <- factor(scores_long_df$model, levels = intersect(desired_levels, unique(scores_long_df$model)))
  
  # Order students
  scores_long_df$name <- factor(scores_long_df$name, levels = sort(unique(scores_long_df$name)))
  
  p <- ggplot(scores_long_df, aes(x = name, y = score, fill = model)) +
    geom_col(position = "stack") +
    labs(
      title = "Stacked Score Contributions per Student",
      x = "Student Name",
      y = "Total Score (Sum across models)", # Y-axis label seems appropriate
      fill = "Grading Model" # Updated Legend Title
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1)
    )
  
  # Save, encode, and return base64 URI
  temp_plot_file <- tempfile(fileext = ".png")
  ggsave(temp_plot_file, plot = p, width = max(6, length(unique(scores_long_df$name))*0.6), height = 5, units = "in", dpi = 96) # Adjusted size calculation
  
  if (file.exists(temp_plot_file)) {
    img_data <- readBin(temp_plot_file, "raw", file.info(temp_plot_file)$size)
    base64_img <- base64enc::dataURI(data = img_data, mime = "image/png")
    unlink(temp_plot_file)
    return(base64_img)
  } else {
    warning("Failed to save the stacked bar plot image.")
    return(NULL)
  }
}

# --- Function to Generate Score Scatterplot ---
generate_score_plot <- function(scores_df, x_var = "score_p", y_var = "score_q", label_var = "name") {
  # Ensure the necessary columns exist
  if (!all(c(x_var, y_var, label_var) %in% names(scores_df))) {
    warning(paste("Missing columns for plotting. Required:", x_var, y_var, label_var))
    return(NULL)
  }
  if (nrow(scores_df) == 0) {
    warning("No data provided to generate_score_plot.")
    return(NULL)
  }
  
  # Map variable names to descriptive labels for plot
  x_label <- map_score_var_to_name(x_var)
  y_label <- map_score_var_to_name(y_var)
  plot_title <- paste("Comparison:", x_label, "vs", y_label)
  
  p <- ggplot(scores_df, aes_string(x = x_var, y = y_var, label = label_var)) +
    geom_point(aes(color = name), size = 3, show.legend = FALSE) +
    geom_text(vjust = -0.5, size = 3) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey") +
    labs(
      title = plot_title, # Use generated title
      x = x_label,       # Use mapped label
      y = y_label        # Use mapped label
    ) +
    theme_minimal() +
    coord_cartesian(xlim = c(0, 100), ylim = c(0, 100))
  
  # Save plot to a temporary file
  temp_plot_file <- tempfile(fileext = ".png")
  ggsave(temp_plot_file, plot = p, width = 6, height = 5, units = "in", dpi = 96)
  
  # Read the file and encode it
  if (file.exists(temp_plot_file)) {
    img_data <- readBin(temp_plot_file, "raw", file.info(temp_plot_file)$size)
    base64_img <- base64enc::dataURI(data = img_data, mime = "image/png")
    unlink(temp_plot_file)
    return(base64_img)
  } else {
    warning("Failed to save the plot image.")
    return(NULL)
  }
}

# --- Data Storage ---
submissions <- data.frame(
  name = character(),
  student_answer = character(),
  desired_answer = character(),
  question = character(),
  stringsAsFactors = FALSE
)

q_string <- "No question set yet."
k_string <- "No key set yet."

# Helper function for preparing input data (unchanged)
comp_all <- function(submissions_df) {
  if (nrow(submissions_df) == 0) {
    return(data.frame(name=character(), student_answer=character(), desired_answer=character()))
  }
  return(submissions_df[, c("name", "student_answer", "desired_answer")])
}


# --- Helper Function for HTML Page Structure (Unchanged) ---
basic_page <- function(title, ..., message = NULL, message_type = "info") {
  message_html <- ""
  if (!is.null(message)) {
    # Basic styling for messages
    style <- switch(message_type,
                    success = "color: green; border: 1px solid green; padding: 10px; margin-bottom: 15px; background-color: #e6ffe6;",
                    error = "color: red; border: 1px solid red; padding: 10px; margin-bottom: 15px; background-color: #ffe6e6;",
                    "color: blue; border: 1px solid blue; padding: 10px; margin-bottom: 15px; background-color: #e6f7ff;") # Default to info
    message_html <- paste0("<div style='", style, "'>", htmlEscape(message), "</div>")
  }
  
  html_content <- glue(
    '<!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>{title}</title>
      <style>
        body {{ font-family: sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        label {{ display: inline-block; width: 100px; margin-bottom: 5px; }}
        input[type="text"] {{ width: 300px; padding: 5px; margin-bottom: 10px; }}
        input[type="submit"], button {{ padding: 8px 15px; margin-right: 10px; cursor: pointer; }}
        .form-section, .info-section, .data-section {{
          border: 1px solid #ccc; padding: 15px; margin-bottom: 20px; border-radius: 5px; background-color: #f9f9f9;
        }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .current-info {{ background-color: #e6f7ff; padding: 10px; border-radius: 5px; margin-bottom: 15px; }}
        .scores-section {{ background-color: #fffadc; }}
        .plot-section {{ background-color: #f0f0f0; }}
        hr {{ border: 0; height: 1px; background: #ccc; margin-top: 20px; margin-bottom: 20px; }}
      </style>
    </head>
    <body>
      <h1>{title}</h1>
      {message_html}
      (...) <!-- Placeholder for specific page content -->
    </body>
    </html>'
  )
  # Use capture.output and paste to handle the ... arguments correctly within glue
  content_dots <- paste(capture.output(print(...)), collapse = "\n")
  final_html <- sub("\\(\\.{3}\\)", content_dots, html_content)
  return(final_html)
}


# --- STUDENT ENDPOINTS (Unchanged) ---

#* @get /student
#* @serializer html
function(req, res, message = NULL, message_type = "info") {
  student_content <- tagList(
    div(class = "info-section",
        h2("Current Question"),
        p(strong("Question: "), htmlEscape(q_string))
    ),
    div(class = "form-section",
        h2("Submit Your Answer"),
        tags$form(action = "/student/submit", method = "post",
                  tags$label(`for`="name", "Name:"),
                  tags$input(type = "text", id="name", name = "name", required = TRUE), br(), br(),
                  tags$label(`for`="answer", "Answer:"),
                  tags$input(type = "text", id="answer", name = "answer", required = TRUE, size="50"), br(), br(),
                  tags$input(type = "submit", value = "Submit Answer")
        )
    )
  )
  
  html <- basic_page(
    title = "Student Portal",
    student_content,
    message = message,
    message_type = message_type
  )
  return(html)
}

#* @post /student/submit
function(req, res) {
  form_data <- req$body
  name <- form_data$name
  answer <- form_data$answer
  msg <- NULL
  msg_type <- "info"
  
  if (is.null(name) || name == "" || is.null(answer) || answer == "") {
    msg <- "Submission Failed: Name and Answer are required."
    msg_type <- "error"
  } else {
    submissions <<- rbind(submissions, data.frame(
      name = name,
      student_answer = answer,
      desired_answer = k_string, # Capture key at submission time
      question = q_string,       # Capture question at submission time
      stringsAsFactors = FALSE
    ))
    msg <- "Submission Accepted Successfully!"
    msg_type <- "success"
    print(submissions) # Console debug
  }
  
  redirect_url <- paste0("/student?message=", URLencode(msg), "&message_type=", msg_type)
  res$status <- 303
  res$setHeader("Location", redirect_url)
  res$body <- ""
  return(res)
}

# --- TEACHER ENDPOINTS ---

#* Generates HTML table from a data frame (Unchanged helper)
#* @param df Data frame to convert
#* @param class Optional CSS class for the table
render_table <- function(df, class = "") {
  if (is.null(df) || nrow(df) == 0) {
    return(p("No data available."))
  }
  header <- tags$tr(lapply(names(df), tags$th))
  rows <- apply(df, 1, function(row) {
    tags$tr(lapply(row, function(cell) tags$td(htmlEscape(cell))))
  })
  tags$table(class = class, tags$thead(header), tags$tbody(rows))
}

#* @get /teacher
#* @serializer html
function(req, res, message = NULL, message_type = "info") {
  # GET request just shows the current state and forms
  teacher_content <- tagList(
    div(class = "info-section current-info",
        h2("Current Active Question & Key"),
        p(strong("Question: "), htmlEscape(q_string)),
        p(strong("Answer Key: "), htmlEscape(k_string))
    ),
    div(class = "form-section",
        h2("Set Question and Answer Key"),
        tags$form(action = "/teacher/submit", method = "post",
                  tags$label(`for`="qn", "New Question:"),
                  tags$input(type = "text", id="qn", name = "qn", size="50"), br(), br(),
                  tags$label(`for`="key", "New Answer Key:"),
                  tags$input(type = "text", id="key", name = "key", size="50"), br(), br(),
                  tags$button(type = "submit", name = "action", value = "set_question", "Set Question & Key")
        )
    ),
    div(class = "data-section",
        h2("Student Submissions"),
        render_table(submissions, class = "submission-table"), # Display raw submissions
        hr(),
        h3("Get Scores for Current Submissions"),
        p("Clicking 'Get Scores' will grade the submissions above based on the ", strong("currently set answer key"), " using multiple methods."),
        tags$form(action = "/teacher/submit", method = "post",
                  tags$button(type = "submit", name = "action", value = "get_scores", "Get Scores")
        )
    )
  )
  
  html <- basic_page(
    title = "Teacher Portal",
    teacher_content,
    message = message,
    message_type = message_type
  )
  return(html)
}


#* @post /teacher/submit
# NO SERIALIZER NEEDED HERE - Returns HTML directly
function(req, res) {
  print("--- Entering /teacher/submit POST endpoint ---")
  
  form_data <- req$body
  action <- form_data$action
  message <- NULL
  message_type <- "info"
  scores_combined_df <- NULL # Original wide format for internal use
  scores_display_df <- NULL  # Renamed wide format for display table
  scores_long_df <- NULL     # Long format for plots
  plot_scatter_uri <- NULL   # Scatter plot
  plot_heatmap_uri <- NULL   # Heatmap plot
  plot_stacked_uri <- NULL   # Stacked bar plot
  
  print(paste("Action received:", ifelse(is.null(action), "NULL", action)))
  if (is.null(action)) { action <- "unknown" }
  
  # --- Action Handling ---
  
  # --- Set Question Action ---
  if (action == "set_question") {
    print("--- Handling 'set_question' action ---")
    new_q <- form_data$qn
    new_k <- form_data$key
    if (!is.null(new_q) && new_q != "" && !is.null(new_k) && new_k != "") {
      q_string <<- new_q
      k_string <<- new_k
      message <- "Question and Answer Key updated successfully!"
      message_type <- "success"
    } else {
      message <- "Failed to update: Both Question and Key must be provided."
      message_type <- "error"
    }
    # Redirect back to GET /teacher with message
    redirect_url <- paste0("/teacher?message=", URLencode(message), "&message_type=", message_type)
    res$status <- 303
    res$setHeader("Location", redirect_url)
    res$body <- ""
    print("--- Redirecting after set_question ---")
    return(res)
    
    # --- Get Scores Action ---
  } else if (action == "get_scores") {
    print("--- Handling 'get_scores' action ---")
    
    # Check conditions for scoring
    if (nrow(submissions) == 0) {
      print("Condition met: No submissions.")
      message <- "No submissions available to score."
      message_type <- "info"
    } else if (k_string == "No key set yet." || k_string == "") {
      print("Condition met: No key set.")
      message <- "Cannot score: Please set an Answer Key first."
      message_type <- "error"
    } else {
      print("Conditions met: Proceeding with scoring.")
      # Use current key for all submissions being scored now
      submissions_to_score <- submissions
      submissions_to_score$desired_answer <- k_string 
      
      print("--- Entering tryCatch block for scoring and plotting ---")
      tryCatch({
        print("tryCatch: Calling comp_all...")
        test_inp <- comp_all(submissions_to_score)
        print(paste("tryCatch: comp_all returned object with", nrow(test_inp), "rows."))
        
        if(nrow(test_inp) > 0) {
          print("tryCatch: Calling prediction functions...")
          # Internal names remain score_p, score_q, score_r
          scores_p <- pred(test_inp, model_p); names(scores_p)[2] <- "score_p"
          scores_q <- pred_q(test_inp); names(scores_q)[2] <- "score_q"
          scores_r <- pred_r(test_inp); names(scores_r)[2] <- "score_r"
          print("tryCatch: Prediction functions finished.")
          
          print("tryCatch: Merging scores (wide format)...")
          all_names <- data.frame(name = unique(test_inp$name))
          scores_merged <- merge(all_names, scores_p, by = "name", all.x = TRUE)
          scores_merged <- merge(scores_merged, scores_q, by = "name", all.x = TRUE)
          scores_combined_df <- merge(scores_merged, scores_r, by = "name", all.x = TRUE) # Internal df
          print(paste("tryCatch: Merged wide data frame has", nrow(scores_combined_df), "rows."))
          
          # Create renamed version for display table
          if (nrow(scores_combined_df) > 0) {
            print("tryCatch: Renaming columns for display table...")
            scores_display_df <- scores_combined_df %>%
              rename(
                `Vector Space Score` = score_p,
                `Keyword Extraction Score` = score_q,
                `N-gram Overlap Score` = score_r
              )
          } else {
            scores_display_df <- data.frame(name=character(), `Vector Space Score`=numeric(), `Keyword Extraction Score`=numeric(), `N-gram Overlap Score`=numeric())
            names(scores_display_df) <- c("name", "Vector Space Score", "Keyword Extraction Score", "N-gram Overlap Score") # Ensure names correct even if empty
          }
          
          
          # Create long format for plots, mapping model codes to names
          if (nrow(scores_combined_df) > 0) {
            print("tryCatch: Pivoting data to long format and mapping names...")
            scores_long_df <- scores_combined_df %>%
              pivot_longer(
                cols = starts_with("score_"),
                names_to = "model_code", # Keep original code temporarily
                names_prefix = "score_",
                values_to = "score"
              ) %>%
              mutate(
                model = map_model_code_to_name(toupper(model_code)) # Map P/Q/R to full names
              ) %>%
              select(-model_code) # Remove the temporary code column
            
            # Set factor levels for consistent plot ordering
            desired_levels <- c("Vector Space Model", "Keyword Extraction Model", "N-gram Overlap Model")
            scores_long_df$model <- factor(scores_long_df$model, levels = intersect(desired_levels, unique(scores_long_df$model)))
            
            print(paste("tryCatch: Pivoted long data frame has", nrow(scores_long_df), "rows with mapped model names."))
          } else {
            scores_long_df <- data.frame(name=character(), model=character(), score=numeric())
            print("tryCatch: No rows in wide data, created empty long data frame.")
          }
          
          
          # --- Generate Plots ---
          print("tryCatch: Generating plots...")
          # Scatter Plot (Model P vs Q by default)
          if (nrow(scores_combined_df) > 0 && "score_p" %in% names(scores_combined_df) && "score_q" %in% names(scores_combined_df)) {
            print("tryCatch:   Generating scatter plot...")
            plot_scatter_uri <- generate_score_plot(scores_combined_df, x_var = "score_p", y_var = "score_q", label_var = "name")
            print(paste("tryCatch:   Scatter plot URI generated:", !is.null(plot_scatter_uri)))
          }
          
          # Heatmap and Stacked Bar (require long format)
          if (!is.null(scores_long_df) && nrow(scores_long_df) > 0) {
            print("tryCatch:   Generating heatmap plot...")
            plot_heatmap_uri <- generate_score_heatmap(scores_long_df)
            print(paste("tryCatch:   Heatmap plot URI generated:", !is.null(plot_heatmap_uri)))
            
            print("tryCatch:   Generating stacked bar plot...")
            plot_stacked_uri <- generate_stacked_bar_plot(scores_long_df)
            print(paste("tryCatch:   Stacked bar plot URI generated:", !is.null(plot_stacked_uri)))
          }
          print("tryCatch: Plot generation finished.")
          
          # Set final message based on success
          if (!is.null(scores_display_df) && nrow(scores_display_df) > 0) {
            message <- "Scores calculated successfully."
            message_type <- "success"
            if (is.null(plot_heatmap_uri) || is.null(plot_stacked_uri) || is.null(plot_scatter_uri)) {
              message <- paste(message, "(One or more plots failed to generate.)")
              message_type <- "warning"
            }
          } else {
            message <- "Scoring completed, but no results were generated."
            message_type <- "warning"
          }
          
        } else { # If test_inp was empty
          message <- "Scoring preparation failed (no valid input data)."
          message_type <- "warning"
        }
        print(paste("tryCatch: Finished successfully. Message:", message))
        
      }, error = function(e) {
        print("--- ERROR HANDLER ENTERED during scoring/plotting ---")
        error_msg <- conditionMessage(e)
        full_error_details <- paste("Error during scoring/plotting:", error_msg)
        print(full_error_details) # Print concise error to console
        # Optionally print traceback() here for more detail in console if needed
        
        message <<- paste("Internal error occurred during scoring or plotting. Check R console logs for details. Error:", error_msg)
        message_type <<- "error"
        # Clear potentially partial results
        scores_combined_df <<- NULL; scores_display_df <<- NULL; scores_long_df <<- NULL; plot_scatter_uri <<- NULL; plot_heatmap_uri <<- NULL; plot_stacked_uri <<- NULL
      }) # End of tryCatch
      print("--- Exited tryCatch block ---")
    } # End of main scoring 'else' block
    
    # --- Render HTML Response Page (always happens after 'get_scores') ---
    print("--- Starting HTML rendering for response ---")
    
    # Define the title for the scatter plot dynamically
    scatter_plot_title <- "Score Comparison Plot" # Default title
    if (!is.null(plot_scatter_uri)) {
      # Assuming scatter plot always compares P vs Q for now
      x_label_for_title <- map_score_var_to_name("score_p") 
      y_label_for_title <- map_score_var_to_name("score_q")
      scatter_plot_title <- paste("Score Comparison (", x_label_for_title, " vs ", y_label_for_title, ")")
    }
    
    teacher_content_response <- tagList(
      # Section 1: Current Q&A (Repeated for context)
      div(class = "info-section current-info", 
          h2("Current Active Question & Key"), 
          p(strong("Question: "), htmlEscape(q_string)), 
          p(strong("Answer Key: "), htmlEscape(k_string))),
      # Section 2: Form to Set Q&A (Repeated for usability)
      div(class = "form-section", 
          h2("Set Question and Answer Key"), 
          tags$form(action = "/teacher/submit", method = "post", 
                    tags$label(`for`="qn", "New Question:"), tags$input(type = "text", id="qn", name = "qn", size="50"), br(), br(), 
                    tags$label(`for`="key", "New Answer Key:"), tags$input(type = "text", id="key", name = "key", size="50"), br(), br(), 
                    tags$button(type = "submit", name = "action", value = "set_question", "Set Question & Key"))),
      # Section 3: Submissions List (Repeated for context)
      div(class = "data-section", 
          h2("Student Submissions"), 
          render_table(submissions, class = "submission-table"), # Display raw submissions
          hr(), 
          h3("Get Scores for Current Submissions"), 
          p("Clicking 'Get Scores' will grade the submissions above based on the ", strong("currently set answer key"), " using multiple methods."), 
          tags$form(action = "/teacher/submit", method = "post", 
                    tags$button(type = "submit", name = "action", value = "get_scores", "Get Scores"))),
      
      # --- Dynamic Sections based on Scoring Results ---
      # Section 4: Scores Table
      if (!is.null(scores_display_df) && nrow(scores_display_df) > 0) { 
        tagList( div(class = "data-section scores-section", 
                     h2("Calculated Scores"), 
                     render_table(scores_display_df, class = "scores-table"))) 
      } else if (action == "get_scores" && message_type != "error") { # Show message if score attempted but no table
        tagList( div(class = "data-section scores-section", h2("Calculated Scores"), p("No scores to display.")))
      } else { "" }, # Don't show section if not attempted or error occurred before table gen
      
      # Section 5: Heatmap Plot
      if (!is.null(plot_heatmap_uri)) { 
        tagList( div(class = "data-section plot-section", 
                     h2("Score Heatmap"), 
                     tags$img(src = plot_heatmap_uri, alt = "Score Heatmap Plot", style = "max-width: 100%; height: auto;"))) 
      } else { "" },
      
      # Section 6: Stacked Bar Plot
      if (!is.null(plot_stacked_uri)) { 
        tagList( div(class = "data-section plot-section", 
                     h2("Stacked Scores per Student"), 
                     tags$img(src = plot_stacked_uri, alt = "Stacked Bar Plot", style = "max-width: 100%; height: auto;"))) 
      } else { "" },
      
      # Section 7: Scatter Plot
      if (!is.null(plot_scatter_uri)) { 
        tagList( div(class = "data-section plot-section", 
                     h2(scatter_plot_title), # Use dynamic title
                     tags$img(src = plot_scatter_uri, alt = "Score Scatter Plot", style = "max-width: 100%; height: auto;"))) 
      } else { "" }
    ) # End of teacher_content_response tagList
    
    print("--- Generating basic page for response ---")
    html_body <- basic_page(
      title = "Teacher Portal - Results", # Updated title for response page
      teacher_content_response,
      message = message, # Display status message from scoring attempt
      message_type = message_type
    )
    
    print("--- Setting response headers and body ---")
    res$setHeader("Content-Type", "text/html")
    res$body <- html_body
    print("--- Returning response from /teacher/submit (get_scores) ---")
    return(res) # Return the fully rendered HTML page
    
    # --- Unknown Action ---
  } else {
    print(paste("--- Handling 'unknown' action:", action, "---"))
    message <- "Unknown form action submitted."
    message_type <- "error"
    redirect_url <- paste0("/teacher?message=", URLencode(message), "&message_type=", message_type)
    res$status <- 303
    res$setHeader("Location", redirect_url)
    res$body <- ""
    print("--- Redirecting after unknown action ---")
    return(res)
  }
}


#* @get /api/submissions
#* @serializer json
function(){
  # Simply return the current state of submissions
  submissions
}