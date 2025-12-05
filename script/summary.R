library(tidyverse)
library(ggplot2)
library(gt)
library(officer)
library(flextable)
library(arrow)

problem_2021_finance <- read_csv("data/problem1_2021_finance.csv")
write_parquet(problem_2021_finance, "data/problem1_2021_finance.parquet")
financial <- read_csv("data/Data_results_ollama/classified_financial_results_12b.csv")
 

tabla_financial <- financial |> 
  count(predicted_financial, name = "n") |> 
  mutate(
    predicted_financial = recode(predicted_financial,
                                 "FINANCIAL"     = "Financial",
                                 "NOT_FINANCIAL" = "Non financial"),
    Prop = round(n / sum(n), 3)
  )

tabla_financial_gt <- tabla_financial |>
  gt() |>
  tab_header(
    title = "Financial Classification Summary"
  ) |>
  fmt_number(
    columns = n,
    sep_mark = ",",
    decimals = 0
  ) |>
  fmt_percent(
    columns = Prop,
    decimals = 1
  ) |>
  tab_source_note(
    source_note = "Nota: Las etiquetas fueron estandarizadas a 'Financial' y 'Non financial'."
  )

# --- Convertir a flextable ---
ft <- flextable(tabla_financial)

# Formato: separador de miles y porcentaje bonito
ft <- colformat_num(ft, col = "n", big.mark = ",", digits = 0)
ft <- colformat_num(ft, col = "Prop", digits = 3)

# --- Agregar nota al pie ---
ft <- add_footer_lines(
  ft,
  values = "Nota: Las etiquetas fueron estandarizadas a 'Financial' y 'Non financial'."
)

# --- Exportar a Word ---
doc <- read_docx()
doc <- body_add_flextable(doc, ft)
print(doc, target = "tabla_financial.docx")


topic_info <-read_csv("output_optimized/topic_info.csv")
topic_results <- read_csv("output_optimized/topic_results.csv")

final_data_financial_with_topics <- topic_results|>
  left_join(topic_info, join_by(topic == Topic) )

write.csv(final_data_financial_with_topics, "data/Data_results_ollama/final_data_financial_with_label.csv")


tabla_summary <- final_data_financial_with_topics|>
  group_by(label)|>
  summarise(n_class= n())|> 
  ungroup()


#non financial sample

financial_sample <- read_csv("data/Data_results_ollama/classified_financial_results_12b.csv") |>
  filter(predicted_financial == "NOT_FINANCIAL") |>
  group_by(label_name) |>
  slice_sample(n = 1)|>
  ungroup()|>
  slice_sample(n=33)
 
fintech <- read_csv("data/Training data/33_assured_fintech.csv")|>
  select(-c(verification_date))

training_data <- bind_rows(financial_sample, fintech)

write_parquet(training_data, "data/Training data/training_data.parquet")

clasi <- read_parquet("data/problem1_2021_finance_classified.parquet")

write.csv(clasi, "data/Training_data/fintech.csv")

