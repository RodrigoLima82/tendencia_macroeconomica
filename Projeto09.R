# --------------------------------------------------------------------
# 01 - Iniciando o script de Machine Learning
# --------------------------------------------------------------------

# Carregando os Pacotes
library(rhdf5)

# --------------------------------------------------------------------
# Carregando o Dataset
data <- h5read('data/train.h5','train')
colnames = data[['axis0']]
data = cbind(t(data[['block0_values']]),t(data[['block1_values']]))
df = as.data.frame(data)
colnames(df) = colnames
rm(colnames)
rm(data)

# Visualizando os dados
head(df)
View(df)

# Visualizando os nomes das colunas
names(df)

# Observacoes Iniciais
# --------------------------------------------------------------------
# O dataset contém 1.710.756 observacoes e 111 atributos
# 01 coluna 'id'
# 01 coluna 'timestamp'
# 05 colunas 'derived_X', onde X é uma sequencia numerica iniciando de 0
# 63 colunas 'fundamental_X', onde X é uma sequencia numerica iniciando de 0 (exceto 'fundamental_4' ???)
# 40 colunas 'technical_X', onde X é uma sequencia numerica iniciando de 0 (exceto _4, _8, _15, _23 e _26 ???)
# 01 coluna "y" que é o alvo


# --------------------------------------------------------------------
# 02 - Analise Exploratoria de Dados
# --------------------------------------------------------------------

# Carregando os Pacotes
library(dplyr)
library(tidyr)
library(ggplot2)
library(Hmisc)
library(corrplot)

# Verificar se existem valores ausentes (missing) em cada coluna
# Valor missing encontrado
any(is.na(df))

# --------------------------------------------------------------------
# Visualizando valores missing no dataset
# Gerando um novo df agrupando os dados missing
missing.values <- df %>%
  gather(key = "key", value = "val") %>%
  mutate(is.missing = is.na(val)) %>%
  group_by(key, is.missing) %>%
  summarise(num.missing = n()) %>%
  filter(is.missing==T) %>%
  select(-is.missing) %>%
  arrange(desc(num.missing)) 

# Visualizando usando ggplot
missing.values %>%
  ggplot() +
  geom_bar(aes(x=key, y=num.missing), stat = 'identity') +
  labs(x='variable', y="number of missing values", title='Number of missing values') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

head(missing.values)

# Observacoes sobre Missing Values
# --------------------------------------------------------------------
# Coluna 3 a 111 sao features independentes
# Existem muitas colunas com valores missing
# Variaveis com maior % de valores missing:
# fundamental_5, fundamental_38, fundamental_6, fundamental_1, fundamental_61 e fundamental_28

# Tratamento dos valores missing
# Aplicando o valor médio das colunas
# --------------------------------------------------------------------
# Criando uma copia do dataset original
new_df <- cbind(df)

for(i in 1:ncol(new_df)){
  new_df[is.na(new_df[,i]), i] <- mean(new_df[,i], na.rm = TRUE)
}

# Verificar se existem valores ausentes (missing) em cada coluna no novo dataset
any(is.na(new_df))


# Avaliando a variavel target "y" (considerando o novo dataset sem valores missing)
# --------------------------------------------------------------------
hist(new_df$y, "FD", xlab="y", ylab="Frequencia")

# Analise estatistica da variavel target 'y'
describe(new_df$y)

# Verificar os outliers
outliers <- distinct(filter(new_df, new_df$y >= 0.0934 | new_df$y <= -0.0860))
describe(outliers$id)

# Observacoes da variavel target
# --------------------------------------------------------------------
# A variavel y mostra uma distribuicao normal, exceto na extremidade da cauda
# Olhando os dados da cauda, verificamos que o outlier gira em torno de 0,0934 e -0,086
# Da a "impressao" que houve corte nos dados
# Verificando os outliers dos IDs, percebe que a maioria sao valores extremos
#   ex: 1284 id distintos onde 95% em 2054


# --------------------------------------------------------------------
# 03 - Feature Engineering
# --------------------------------------------------------------------

# Carregando os Pacotes
library(scales)

# Gerando uma copia do dataset
dfTrain <- new_df

# Normalizando as variáveis numericas
scale.features <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- scale(df[[variable]], center=T, scale=T)
  }
  return(df)
}

# Normalizando todas as variaveis numericas
numeric.vars <- unlist(lapply(dfTrain, is.numeric)) 
dfTrain <- scale.features(dfTrain, numeric.vars)

# Analise de Correlacao
# --------------------------------------------------------------------

# Separando as colunas numericas para correlacao
numeric.vars <- unlist(lapply(dfTrain, is.numeric)) 
data_cor <- cor(dfTrain[,numeric.vars])

# Visualizando a correlacao de uma amostra do dataset
rrow <- sample(1:dim(dfTrain)[1],50)
corGraph <- cor(dfTrain[rrow, ])
corrplot(corGraph, order = "FPC", method = "number", type = "lower", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0),number.cex = 0.7, number.digits = 2)


# Observacoes da correlacao
# --------------------------------------------------------------------
# De acordo com o grafico de correlacao, as features abaixo já estao correlacionadas cmo outras features existentes
# Por isso, vou remove-las no momento certo, na avaliacao de alguns modelos
# fundamental_61,fundamental_11,fundamental_56,fundamental_26,fundamental_10,fundamental_15,
# fundamental_57,fundamental_41,fundamental_30,fundamental_53,fundamental_42,fundamental_26,
# fundamental_10,fundamental_60,fundamental_48,fundamental_55,fundamental_11,fundamental_45,
# fundamental_16,fundamental_34,fundamental_12,fundamental_51,fundamental_43,fundamental_1,
# fundamental_42,fundamental_30,fundamental_53


# --------------------------------------------------------------------
# 05 - Criando alguns modelos de ML para comparacoes
# --------------------------------------------------------------------

# Observacoes
# --------------------------------------------------------------------
# Como a variavel target 'y' é uma variável contínua, o modelo mais fácil de construir é o modelo de regressão linear.
# Vou criar três modelos de regressão: linear regression, ridge regression and lasso regression. 
# A diferença entre ridge e lasso está na função de penalidade. 
# Também vou avaliar a performance usando o modelo Generalized Boosted Regression Modeling (GBM) e
# eXtreme Gradient Boosting (XGBoost)

# Carregando os Pacotes
library(caret)

# Gerando dados de treino e de teste
# --------------------------------------------------------------------
splits <- createDataPartition(dfTrain$y, p=0.8, list=FALSE)

# Separando os dados de treino e teste
dados_treino <- dfTrain[ splits,]
dados_teste <- dfTrain[-splits,]

# Definindo a formula para os modelos
# Usando as colunas com maiores correlacoes
#formula <- "y ~ ."
formula <- "y ~ fundamental_5 + fundamental_7 + fundamental_14 + 
                 fundamental_17 + fundamental_19 + fundamental_20 + fundamental_21 + 
                 fundamental_31 + fundamental_33 + fundamental_36 + fundamental_40 + 
                 fundamental_44 + fundamental_46 + fundamental_47 + fundamental_49 + 
                 fundamental_63 + technical_0 + technical_7 + technical_11 + 
                 technical_16 + technical_20 + technical_21 + technical_22 + 
                 technical_27 + technical_30 + technical_32 + technical_36 + 
                 technical_37 + technical_44"
formula <- as.formula(formula)


# Funcao para calcular o valor R (que sera a forma de avaliacao dos modelos)
# Usei este metodo pra ficar de acordo com a solicitacao do desafio no Kaggle
r_value <- function(R_sq){
  R_val <- sign(R_sq)*sqrt(abs(R_sq))
  return(R_val)
}

# Construindo um modelo Linear Regression (LM)
# --------------------------------------------------------------------
set.seed(1234)
controlLM <- trainControl(method="cv", number=5)
modeloLM <- train(formula, data = dados_treino, method = "lm", trControl=controlLM)

# Fazendo previsoes para avaliar a performance do modelo nos dados de teste
predLM <- modeloLM %>% predict(dados_teste)

# Avaliando a performance do modelo
linearR = r_value(R2(predLM, dados_teste$y))
print(linearR)

# Construindo um modelo Ridge Regression
# --------------------------------------------------------------------
# Criando o modelo
set.seed(1234)
controlRidge <- trainControl(method="cv", number=5)
modeloRidge <- train(formula, data = dados_treino, method = "glmnet", 
                     trControl = controlRidge, 
                     tuneGrid = expand.grid(alpha = 0, lambda = 0))

# Fazendo previsoes para avaliar a performance do modelo
predRigde <- modeloRidge %>% predict(dados_teste)

# Avaliando a performance do modelo
ridgeR = r_value(R2(predRigde, dados_teste$y))
print(ridgeR)

# Construindo um modelo Lasso Regression
# --------------------------------------------------------------------
# Criando o modelo
set.seed(1234)
controlLasso <- trainControl(method="cv", number=5)
modeloLasso <- train(formula, data = dados_treino, method = "glmnet", 
                     trControl = controlLasso, 
                     tuneGrid = expand.grid(alpha = 1, lambda = 0))

# Fazendo previsoes para avaliar a performance do modelo
predLasso <- modeloLasso %>% predict(dados_teste)

# Avaliando a performance do modelo
lassoR = r_value(R2(predLasso, dados_teste$y))
print(lassoR)

# Construindo um modelo Generalized Boosted Regression Modeling (GBM)
# --------------------------------------------------------------------
# Criando o modelo
set.seed(1234)
controlGBM <- trainControl(method="cv", number=2)
modeloGBM <- train(formula, data=dados_treino, method="gbm",  verbose=FALSE, trControl=controlGBM)

# Fazendo previsoes para avaliar a performance do modelo
predGBM <- modeloGBM %>% predict(dados_teste)

# Avaliando a performance do modelo
GBM_R = r_value(R2(predGBM, dados_teste$y))
print(GBM_R)

# Comparando a performance dos modelos
# --------------------------------------------------------------------
p <- data.frame(linearR, ridgeR, lassoR, GBM_R)
colnames(p) = c('Linear', 'Ridge', 'Lasso', 'GBM')
m <- as.matrix(p)
m

# Observacoes da Performance dos modelos (dados de teste)
# --------------------------------------------------------------------
# Podemos verificar que usando todas as variaveis o modelo Generalized Boosted Regression Modeling teve o melhor desempenho
# Vou usar esse modelo para otimizacao dos hyperparametros

# --------------------------------------------------------------------
# 06 - Otimizando o modelo Generalized Boosted Regression Modeling
# --------------------------------------------------------------------

# Carregando os Pacotes
library(gbm)

# Criando grid de hyperparameter
hyper_grid <- expand.grid(
  shrinkage = c(.01, .1),
  interaction.depth = c(1, 3),
  n.minobsinnode = c(5, 10),
  bag.fraction = c(.65, .8), 
  optimal_trees = 0,
  min_RMSE = 0
)

# Numero total de combinacoes
nrow(hyper_grid)

# Grid Search 
for(i in 1:nrow(hyper_grid)) {

  # Criando o modelo  
  set.seed(1234)

  gbm.tune <- gbm(
    formula = formula,
    distribution = "gaussian",
    data = dados_treino,
    n.trees = 100,
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i],
    train.fraction = .75,
    n.cores = NULL,
    verbose = FALSE
  )
  
  # Verificando os erros do treinamento
  hyper_grid$optimal_trees[i] <- which.min(gbm.tune$valid.error)
  hyper_grid$min_RMSE[i] <- sqrt(min(gbm.tune$valid.error))
}

hyper_grid %>% 
  dplyr::arrange(min_RMSE) %>%
  head(10)

# print results
print(gbm.tune)

# --------------------------------------------------------------------
# 07 - Avaliando o modelo otimizado nos dados de teste
# --------------------------------------------------------------------

# train GBM model
modeloGBM_otm <- gbm(
  formula = formula,
  distribution = "gaussian",
  data = dados_treino,
  n.trees = 80,
  interaction.depth = 3,
  shrinkage = 0.1,
  n.minobsinnode = 10,
  bag.fraction = 0.8, 
  train.fraction = 1,
  n.cores = NULL,
  verbose = FALSE
) 

par(mar = c(5, 8, 1, 1))
summary(
  modeloGBM_otm, 
  cBars = 10,
  method = relative.influence, # also can use permutation.test.gbm
  las = 2
)

# predict values for test data
predGBM_otm <- predict(modeloGBM_otm, n.trees = modeloGBM_otm$n.trees, dados_teste)

# Avaliando a performance do modelo
GBM_otm = r_value(R2(predGBM_otm, dados_teste$y))
print(GBM_otm)

# Comparando a performance dos modelos finais
# --------------------------------------------------------------------
p <- data.frame(linearR, ridgeR, lassoR, GBM_R, GBM_otm)
colnames(p) = c('Linear', 'Ridge', 'Lasso', 'GBM', 'GBM_otm')
m <- as.matrix(p)
m

# --------------------------------------------------------------------
# 09 - Conclusao Final
# --------------------------------------------------------------------

# De acordo com o Kaggle, foi descrito para não desanimar com baixos valores de R
# Em finanças, dada a alta proporção de sinal-ruído, até um pequeno R pode oferecer um valor significativo!
# --------------------------------------------------------------------
# O melhor algoritmo para esse dataset é foi o Generalized Boosted Regression Modeling
# O modelo GBM considerando a selecao das variaveis do dataset teve uma performance de 0.03263341
# Otimizando este modelo, obtive uma performance de 0.03726622
# Comparando com o leaderboard do Kaggle, foi um resultado muito bom (apesar de nao submeter)
# O ideal seria validar esse modelo gerando o submission, mas não é mais possivel pois a competição foi encerrada