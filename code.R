###1. IMPORTACIÓN DE LOS DATOS###

#1.1 Carga de paquetes necesarios para el proyecto
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(ggthemr)) install.packages("ggthemr", repos = "http://cran.us.r-project.org")
if(!require(readxl)) install.packages("readxl", repos = "http://cran.us.r-project.org")
if(!require(openxlsx)) install.packages("openxlsx", repos = "http://cran.us.r-project.org")
if(!require(tm)) install.packages("tm", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(ggthemr)
library(readxl)
library(openxlsx)
library(tm)
library(e1071)
library(caret)


#1.2 Fijamos directorio de trabajo e importamos el conjunto de datos
#https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download
setwd("C:/Users/adria/Desktop/PROYECTOS/spam_classifier - R")
data <- read.csv("spam_ham.csv", stringsAsFactors = FALSE)

#1.3 Eliminamos residuos en columnas 3,4 y 5, asignamos nombre a las columnas y cambiamos el tipo de dato de la variable dependiente
data <- data[,1:2]
colnames(data) <- c("tipo","sms")
data <- data %>% 
  mutate(tipo = as.factor(tipo))

#1.4 Visualizamos la proporcion del tipo de SMS
ggthemr("flat")
data %>% 
  ggplot(mapping = aes(tipo, fill = tipo)) + 
  geom_bar() + 
  ggtitle("SMS en función de su tipo") +
  xlab("Tipo") + 
  ylab("Proporción") + 
  theme(legend.position = "none") + 
  scale_fill_manual(values = c(ham = "orange", spam = "black")) 

###2. PREPROCESAMIENTO DE LOS DATOS###

#2.1 Creamos un corpus con los sms y procedemos a su "limpieza" por pasos
sms_corpus <- VCorpus(VectorSource(data$sms))
sms_corpus_clean <- sms_corpus %>% #eliminamos mayusculas
  tm_map(content_transformer(tolower))
sms_corpus_clean <- sms_corpus_clean %>% #eliminamos numeros
  tm_map(removeNumbers)
sms_corpus_clean <- sms_corpus_clean %>% #eliminamos stopwords
  tm_map(removeWords,stopwords())
sms_corpus_clean <- sms_corpus_clean %>% #eliminamos signos de puntuacion
  tm_map(removePunctuation)
sms_corpus_clean <- sms_corpus_clean %>% #pasamos a la raiz de las palabras
  tm_map(stemDocument)
sms_corpus_clean <- sms_corpus_clean %>% #eliminamos por último los posibles espacios extra que se hayan generado
  tm_map(stripWhitespace)
tibble(before_clean = c(as.character(sms_corpus[[9]]), #observamos una tabla con los tres primeros mensajes antes y depués de la limpieza
                        as.character(sms_corpus[[11]]),
                        as.character(sms_corpus[[15]])),
       after_clean = c(as.character(sms_corpus_clean[[9]]),
                       as.character(sms_corpus_clean[[11]]),
                       as.character(sms_corpus_clean[[15]])))

#2.2 Pasamos a estructurar el texto en una matriz donde las filas son los sms y las columnas las palabras (tokenización)
sms_matrix <- DocumentTermMatrix(sms_corpus_clean)


#2.3 Pasamos a crear la matriz de regresores y el vector "tipo de SMS" y sus correspondientes particiones
n_entrenamiento <- nrow(sms_matrix) * 0.75
n_testeo <- n_entrenamiento + 1 
train_x <- sms_matrix[1:n_entrenamiento,]
test_x <- sms_matrix[n_testeo:nrow(sms_matrix),]
train_y <- data[1:n_entrenamiento,]$tipo
test_y <- data[n_testeo:nrow(sms_matrix),]$tipo
#comprobamos que la representatividad de las clases en los dos conjuntos (testeo y entrenamiento) coincide
prop.table(table(test_y))
prop.table(table(train_y))

#2.4 Buscamos eliminar regresores (palabras) poco frecuentes en los SMS
freq_words <- findFreqTerms(train_x,5) #nos quedamos con palabras que aparezcan en al menos 5 mensajes
train_x <- train_x[,freq_words]
test_x <- test_x[,freq_words]

#2.5 Cambiamos de matriz numérica a categórica (naive bayes trabaja con categoricas o numericas discretizadas)
convert_counts <- function(x){
  x <- ifelse(x>0,"Yes","No")
}
train_x <- apply(train_x,MARGIN = 2,convert_counts) 
test_x <- apply(test_x,MARGIN = 2,convert_counts) 

###3. ENTRENAMIENTO Y EVALUACIÓN DEL MODELO

#3.1 Entrenamos al algoritmo y generamos las predicciones

model <- naiveBayes(train_x,train_y,laplace = 1) #factor de corrección laplace =1 para las palabras que no aparezcan en una de las dos clases
predictions <- predict(model,test_x,type = "class")


#3.2 Graficamos la matriz de confusion
c_nb <- as.data.frame(confusionMatrix(predictions,test_y)$table)

plotTable <- c_nb %>%
  mutate(goodbad = ifelse(c_nb$Prediction == c_nb$Reference, "good", "bad")) %>%
  group_by(Reference) %>%
  mutate(prop = Freq/sum(Freq))

ggplot(data = plotTable, mapping = aes(x = Reference, y = Prediction, fill = goodbad, alpha = prop)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = .5, fontface  = "bold", alpha = 1) +
  scale_fill_manual(values = c(good = "orange", bad = "black")) +
  xlim(rev(levels(c_nb$Reference))) +
  ggtitle("Confusion Matrix - Naive Bayes",
          subtitle = "Overall accuracy = 97.34%") +
  theme(plot.title = element_text(size=rel(1.5)))
