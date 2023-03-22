library(readxl)
library(ggplot2)
library(tidyverse)
library(randomForest)
library(e1071)
library(caret)
library(MLmetrics)

# Lendo os dados
dados = read_excel("FEV-data-Excel.xlsx")

View(dados)

# Renomeando as colunas para nomes mais entendíveis, e retirando os espaços
col_names = c("Car",
                  "Company",
                  "Model",
                  "Minimal_Price_Gross",
                  "Engine_Power_Km",
                  "Maximum_Torque_Nm",
                  "Type_of_Brakes",
                  "Drive_Type",
                  "Battery_Capacity_Kwh",
                  "Range_WLTP_Km",
                  "Wheelbase_Cm",
                  "Length_Cm",
                  "Width_Cm",
                  "Height_Cm",
                  "Minimal_Empty_Weight_Kg",
                  "Permissable_Gross_Weight_Kg",
                  "Maximun_Load_Capacity_Kg",
                  "Number_of_Seats",
                  "Number_of_Doors",
                  "Tire_Size",
                  "Maximum_Speed_Kph",
                  "Boot_Capacity_Vda",
                  "Acceleration_0_to_100",
                  "Maximum_DC_Charging_Power_Kw",
                  "Mean_Energy_Consumption_Kwh_per_100_Km")

names(dados) = col_names

# Verificando valores NA
colSums(is.na(dados))
dim(dados)

# Decidimos por retirarmos as linhas com valores NA, pois como o dataset não é grande,
# Se fôssemos imputar valores estaríamos distorcendo muito os dados, comprometendo
# O resultado da previsão do modelo
dados = na.omit(dados)
dim(dados)

# Verificando os tipos de dados
str(dados)

# Transformando as variáveis categóricas em fatores
unique(dados$Type_of_Brakes)
dados$Type_of_Brakes = as.factor(dados$Type_of_Brakes)

unique(dados$Drive_Type)
dados$Drive_Type = as.factor(dados$Drive_Type)

unique(dados$Number_of_Seats)
dados$Number_of_Seats = as.factor(dados$Number_of_Seats)

unique(dados$Number_of_Doors)
dados$Number_of_Doors = as.factor(dados$Number_of_Doors)

unique(dados$Tire_Size)
dados$Tire_Size = as.factor(dados$Tire_Size)

str(dados)

summary(dados)

# Criando um dataframe secundário da classe data.frame, ao invés da classe tbl_df
df = as.data.frame(dados)

# Criando boxplot para as variáveis numéricas
lapply(col_names, function(x){
  if(!is.factor(df[,x]) & !is.character(df[,x])) {
    ggplot(df, aes_string(x)) +
      geom_boxplot() +
      ggtitle(paste("Boxplot de",x))}})

# Criando gráfico de barras para as variáveis do tipo fator
lapply(col_names, function(x){
  if(is.factor(df[,x])) {
    ggplot(df, aes_string(x)) +
      geom_bar() +
      ggtitle(paste("Frequência da variável",x))}})

# Vamos verificar as variáveis mais importarntes para a previsão da variável alvo através
# do randomForest

modelo_escolha_variaveis = randomForest(Mean_Energy_Consumption_Kwh_per_100_Km ~ ., data = df,
                          ntree = 100, nodesize = 10, importance = TRUE)

varImpPlot(modelo_escolha_variaveis)

# Vamos construir o modelo apenas com as 11 variáveis mais importantes

var_importantes = c("Minimal_Price_Gross",
                    "Engine_Power_Km",
                    "Maximum_Torque_Nm",
                    "Battery_Capacity_Kwh",
                    "Wheelbase_Cm",
                    "Length_Cm",
                    "Width_Cm",
                    "Minimal_Empty_Weight_Kg",
                    "Permissable_Gross_Weight_Kg",
                    "Maximun_Load_Capacity_Kg",
                    "Car",
                    "Mean_Energy_Consumption_Kwh_per_100_Km")

dados_filtrados = df[, var_importantes]
View(dados_filtrados)

# Criando os dados de treino e de teste

index = createDataPartition(dados_filtrados$Mean_Energy_Consumption_Kwh_per_100_Km, p = .75, list = FALSE)
treino = dados_filtrados[index, ]
teste = dados_filtrados[-index, ]

modelo_v01 = lm(Mean_Energy_Consumption_Kwh_per_100_Km ~ ., data = treino)
predict(modelo_v01, newdata = teste[, -12])

# Erro devido a variável Car ter valores diferentes nos dados de teste
# das existentes nos dados de treino. Faz todo o sentido, afinal estamos
# trabalhando com um dataset bem pequeno. Vamos remover essa variável e
# refazer o processo.

dados_filtrados$Car = NULL
index = createDataPartition(dados_filtrados$Mean_Energy_Consumption_Kwh_per_100_Km, p = .75, list = FALSE)
treino = dados_filtrados[index, ]
teste = dados_filtrados[-index, ]

modelo_v01 = lm(Mean_Energy_Consumption_Kwh_per_100_Km ~ ., data = treino)
predict(modelo_v01, newdata = teste[, -11])
summary(modelo_v01)

previsoes_v01 = data.frame(observado = teste[, 11],
                           previsto = predict(modelo_v01, newdata = teste[, -11]))

# Calculando o R-Squared com o pacote Caret
R2(previsoes_v01$previsto, previsoes_v01$observado)

# R-Squared = 0.73

# Vamos tentar outro algoritmo, o SVM

modelo_v02 = svm(Mean_Energy_Consumption_Kwh_per_100_Km ~ ., data = treino)
predict(modelo_v02, newdata = teste[, -11])
summary(modelo_v02)

previsoes_v02 = data.frame(observado = teste[, 11],
                           previsto = predict(modelo_v02, newdata = teste[, -11]))

R2(previsoes_v02$previsto, previsoes_v02$observado)

# R-Squared = 0.91

# Vamos tentar mais uma vez com o Random Forest

modelo_v03 = randomForest(Mean_Energy_Consumption_Kwh_per_100_Km ~ ., data = treino)
predict(modelo_v03, newdata = teste[, -11])

previsoes_v03 = data.frame(observado = teste[, 11],
                           previsto = predict(modelo_v03, newdata = teste[, -11]))

R2(previsoes_v03$previsto, previsoes_v03$observado)

# R-Squared = 0.84
# Um bom valor, mas inferior ao que obtivemos usando o SVM

# O nosso melhor modelo foi feito com o SVM. Estas são as suas previsões:

View(previsoes_v02)