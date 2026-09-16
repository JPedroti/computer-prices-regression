# Computer Prices — Regression Challenge

## 1. MISSÃO DO PROJETO

Este repositório contém a solução para o **Projeto 03 — Computer Prices Regression Challenge** da Mentoria em Ciência de Dados.

O objetivo é desenvolver uma solução de Machine Learning competitiva, rigorosa, reproduzível e profissional, buscando maximizar a pontuação final da competição.

A prioridade é obter excelente desempenho no **teste secreto**, preservando simultaneamente os pontos de:

- performance no teste secreto;
- ausência de overfitting segundo a regra oficial;
- reprodutibilidade e organização;
- explicabilidade com SHAP.

A solução final deve ser tecnicamente defensável e reproduzível por outra pessoa a partir do repositório.

---

# 2. REGULAMENTO OFICIAL

O regulamento estabelece:

- 55 pontos para RMSE no teste secreto;
- 15 pontos para overfitting;
- 20 pontos para reprodutibilidade e organização;
- 10 pontos para SHAP;
- total de 100 pontos.

Não existe algoritmo obrigatório.

São permitidos, entre outros:

- modelos lineares;
- árvores;
- Random Forest;
- Gradient Boosting;
- XGBoost;
- LightGBM;
- CatBoost;
- ensembles;
- outras abordagens de regressão.

Feature engineering, seleção de variáveis, transformações, encoding e tuning são livres.

Cross-validation é opcional.

O teste secreto nunca deve ser utilizado para treinamento, seleção de features, tuning ou qualquer decisão que introduza leakage.

---

# 3. PRINCÍPIO FUNDAMENTAL

Trate este projeto como uma **competição de Machine Learning para dados tabulares**.

Não basta criar um modelo que funciona.

Precisamos:

1. entender profundamente o dataset;
2. estabelecer baselines;
3. testar várias famílias de modelos;
4. investigar feature engineering;
5. investigar preprocessing;
6. realizar tuning;
7. medir robustez;
8. testar ensembles;
9. avaliar overfitting;
10. garantir reprodutibilidade;
11. produzir SHAP;
12. construir um pipeline de inferência realmente executável sobre dados novos.

Não escolher o primeiro modelo razoável.

Não assumir que o modelo mais complexo é o melhor.

Não assumir que o modelo que apresenta o melhor resultado em uma única divisão será o melhor no teste secreto.

Toda decisão importante deve ser sustentada por evidência experimental.

---

# 4. DATASET

O dataset de desenvolvimento está em:

`data/computer_prices_train_80.csv`

A coluna target é:

`price`

O dataset disponível ao aluno é utilizado para:

- exploração;
- treino;
- validação;
- experimentação;
- seleção do modelo.

O conjunto de teste secreto pertence ao mentor e não deve ser acessado.

## Regra crítica

Nunca:

- procurar;
- inferir;
- reconstruir;
- solicitar;
- tentar acessar;
- tentar explorar

o dataset secreto ou seus targets.

Nunca utilizar informações externas que revelem direta ou indiretamente o resultado do teste secreto.

---

# 5. REGRA DE OURO CONTRA LEAKAGE

Nenhuma técnica de validação, preprocessing ou feature engineering pode utilizar informação do conjunto de validação de forma indevida.

Transformações que aprendem parâmetros a partir dos dados devem ser ajustadas somente no conjunto de treinamento correspondente.

Exemplos:

- imputação;
- scaling;
- encoding;
- seleção de features;
- PCA;
- feature selection;
- target encoding;
- qualquer transformação dependente da distribuição dos dados.

Sempre que apropriado, encapsular essas operações em pipelines.

Evitar qualquer forma de target leakage.

Sempre investigar possíveis duplicatas entre treino e validação.

Investigar cuidadosamente features que possam carregar informação do target direta ou indiretamente.

---

# 6. ESTRUTURA DO PROJETO

A estrutura desejada é:

```text
computer-prices-regression/
│
├── data/
│   └── computer_prices_train_80.csv
│
├── notebooks/
│   └── 01_exploration_and_modeling.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── inference.py
│   └── utils.py
│
├── models/
│   └── final_model.joblib
│
├── experiments/
│   ├── results.csv
│   ├── experiment_log.md
│   └── ...
│
├── reports/
│   ├── figures/
│   └── ...
│
├── tests/
│   └── ...
│
├── configs/
│   └── ...
│
├── requirements.txt
├── README.md
├── .gitignore
└── CLAUDE.md
```

Não crie arquivos ou diretórios desnecessários.

Adapte essa estrutura quando houver uma razão técnica clara para fazê-lo.

---

# 7. PRIMEIRA TAREFA: INSPEÇÃO

Antes de implementar modelos sofisticados, inspecione completamente o repositório e o dataset.

Determine:

- número de linhas;
- número de colunas;
- target;
- tipos de dados;
- missing values;
- cardinalidade;
- duplicatas;
- distribuições;
- outliers;
- relações entre features;
- distribuição do target;
- possíveis transformações;
- features potencialmente redundantes;
- features suspeitas de leakage.

Não invente propriedades do dataset.

Todas as conclusões devem resultar da análise real dos dados.

---

# 8. BASELINE

Criar inicialmente baselines simples.

Avaliar, quando apropriado:

- baseline ingênuo baseado na média/mediana;
- modelo linear;
- Ridge;
- outros modelos simples relevantes.

Os baselines existem para estabelecer referências quantitativas.

Registrar os resultados.

---

# 9. EXPLORAÇÃO DE MODELOS

Investigar sistematicamente modelos apropriados para regressão tabular.

Considerar, quando compatível com os dados:

### Modelos lineares

- Linear Regression
- Ridge
- Lasso
- Elastic Net

### Modelos baseados em árvores

- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- HistGradientBoosting

### Gradient boosting avançado

- XGBoost
- LightGBM
- CatBoost

### Ensembles

- blending;
- weighted averaging;
- stacking.

Não usar uma biblioteca simplesmente por ser popular.

Priorizar métodos compatíveis com as características observadas no dataset.

---

# 10. FEATURE ENGINEERING

Investigar feature engineering de maneira sistemática.

Possibilidades:

- transformações matemáticas;
- log transforms;
- interações;
- agregações;
- razões;
- diferenças;
- decomposição de variáveis;
- tratamento de categóricas;
- encoding;
- discretização quando justificável;
- remoção de features pouco informativas;
- tratamento de alta cardinalidade.

Não criar dezenas de features indiscriminadamente.

Toda nova família de features deve ser avaliada empiricamente.

Evitar feature engineering que introduza leakage.

Registrar quais transformações foram testadas e seus resultados.

---

# 11. PREPROCESSING

O preprocessing deve ser consistente entre:

- treinamento;
- validação;
- inferência.

Sempre que possível, encapsular preprocessing e modelo em um pipeline ou estrutura equivalente.

O objetivo é garantir que:

```text
dados novos
    ↓
mesmo preprocessing
    ↓
modelo final
    ↓
previsão
```

sem etapas manuais.

Nunca depender de alterações manuais quando o mentor fornecer novos dados.

---

# 12. ESTRATÉGIA DE VALIDAÇÃO

Escolha a estratégia de validação com base na estrutura real do dataset.

Avaliar quando apropriado:

- holdout;
- K-Fold;
- Repeated K-Fold;
- outras estratégias adequadas.

O critério principal é obter uma estimativa confiável da generalização.

Sempre que computacionalmente razoável, avaliar estabilidade entre múltiplos folds e/ou seeds.

Registrar:

- RMSE;
- MAE;
- média;
- desvio padrão;
- diferença entre treino e validação;
- estabilidade.

Não fazer tuning excessivo no mesmo validation split.

---

# 13. MÉTRICAS

A métrica principal é:

`RMSE`

A métrica auxiliar é:

`MAE`

Reportar pelo menos:

- RMSE treino;
- RMSE validação;
- MAE treino;
- MAE validação.

Sempre deixar claro quais dados produziram cada métrica.

Nunca comparar métricas obtidas em conjuntos diferentes como se fossem diretamente equivalentes sem explicação.

---

# 14. HYPERPARAMETER TUNING

Fazer tuning de maneira organizada.

Estratégia preferencial:

### Fase 1 — exploração

Explorar regiões amplas dos hiperparâmetros.

### Fase 2 — refinamento

Concentrar a busca nas regiões promissoras.

Não desperdiçar recursos em grids enormes sem justificativa.

Registrar:

- modelo;
- parâmetros;
- seed;
- estratégia de validação;
- RMSE;
- MAE;
- observações.

Sempre preservar o melhor resultado encontrado.

---

# 15. ROBUSTEZ

Um resultado excelente em um único split não é suficiente.

Quando computacionalmente razoável:

- testar múltiplos seeds;
- testar múltiplos folds;
- avaliar dispersão dos resultados;
- verificar estabilidade das features;
- verificar estabilidade dos hiperparâmetros;
- comparar modelos em condições semelhantes.

Um ganho pequeno, porém extremamente instável, deve ser tratado com cautela.

---

# 16. ENSEMBLES

Investigar se combinações dos melhores modelos produzem melhoria.

Possibilidades:

- média simples;
- média ponderada;
- blending;
- stacking.

Comparar quantitativamente ensemble versus melhor modelo individual.

Não adicionar complexidade sem evidência de benefício.

---

# 17. OVERFITTING — REGRA OFICIAL

Implementar exatamente a regra definida pelo projeto.

Procedimento:

1. separar dados em treino e validação;
2. treinar o modelo final somente no treino;
3. calcular RMSE no treino;
4. calcular RMSE na validação;
5. aplicar bootstrap às observações de treino;
6. obter IC 95% do RMSE de treino;
7. aplicar bootstrap separadamente à validação;
8. obter IC 95% do RMSE de validação;
9. comparar os intervalos.

Segundo a regra operacional do desafio:

- ICs se cruzando → não há overfitting → 15 pontos;
- ICs não se cruzando e validação apresentando erro maior → overfitting → 0 pontos.

Essa regra deve ser implementada e documentada exatamente.

---

# 18. BOOTSTRAP

O bootstrap deve ser tecnicamente correto.

Documentar:

- número de reamostragens;
- método utilizado;
- seed;
- estatística calculada;
- forma de obtenção do intervalo;
- resultados.

Não produzir apenas números finais sem deixar claro como foram obtidos.

---

# 19. MODELO FINAL

Somente selecionar o modelo final depois de concluir uma investigação razoavelmente ampla.

O modelo final deve:

- apresentar desempenho competitivo;
- possuir comportamento estável;
- atender ao critério de overfitting quando possível;
- funcionar no pipeline de inferência;
- ser serializável;
- ser carregável sem retreinamento;
- ser compatível com SHAP quando necessário.

Salvar o artefato final, por exemplo:

`models/final_model.joblib`

Se o melhor sistema for composto por múltiplos modelos, salvar o objeto completo necessário para reproduzir exatamente a inferência.

---

# 20. PIPELINE DE INFERÊNCIA

A entrega deve permitir:

```text
CSV novo
   ↓
leitura
   ↓
preprocessing
   ↓
feature engineering
   ↓
modelo final
   ↓
previsões
```

O pipeline deve:

- aceitar CSV/DataFrame com a estrutura das features;
- aplicar automaticamente todas as transformações;
- carregar o modelo salvo;
- gerar uma previsão por linha;
- preservar a ordem das observações;
- produzir uma saída clara.

Não fazer preprocessing manual antes de executar o pipeline.

Não exigir retreinamento.

Não exigir edição do código para executar sobre novos dados equivalentes.

---

# 21. SHAP

A análise SHAP deve ser feita sobre o modelo final entregue.

Obrigatoriamente:

### Global

- visualização apropriada;
- ranking das features;
- interpretação das principais features;
- discussão do sentido dos efeitos quando suportado pela visualização.

### Local

- pelo menos uma observação individual;
- explicação das features que aumentaram a previsão;
- explicação das features que reduziram a previsão.

Nunca tratar SHAP como demonstração de causalidade.

---

# 22. NOTEBOOK

O notebook principal deve contar a história completa do desenvolvimento.

Ordem recomendada:

1. objetivo;
2. carregamento;
3. entendimento dos dados;
4. EDA;
5. preprocessing;
6. baseline;
7. experimentos;
8. comparação dos modelos;
9. tuning;
10. validação;
11. bootstrap;
12. análise de overfitting;
13. seleção do modelo final;
14. SHAP global;
15. SHAP local;
16. conclusão.

O notebook deve executar do início ao fim.

Não esconder lógica importante em células ocultas.

Não depender de estado prévio não documentado.

---

# 23. EXPERIMENT TRACKING

Todo experimento importante deve ser registrado.

No mínimo:

```text
experiment_id
timestamp
model
features
preprocessing
hyperparameters
seed
validation_strategy
train_rmse
validation_rmse
train_mae
validation_mae
notes
```

Usar `experiments/results.csv` ou estrutura equivalente.

O histórico deve permitir responder:

- qual foi o melhor modelo?
- em quais condições?
- qual feature engineering foi usada?
- qual preprocessing foi usado?
- quais hiperparâmetros foram utilizados?
- qual foi o ganho?

---

# 24. REPRODUTIBILIDADE

Fixar seeds quando tecnicamente apropriado.

Documentar dependências.

Criar:

`requirements.txt`

O README deve explicar:

- instalação;
- ambiente;
- estrutura do projeto;
- treinamento;
- avaliação;
- inferência;
- localização do modelo final;
- geração de previsões.

Outra pessoa deve conseguir reproduzir o projeto sem conhecimento prévio do desenvolvimento.

---

# 25. TESTES

Criar testes para componentes críticos quando fizer sentido.

Priorizar:

- preprocessing;
- feature engineering;
- carregamento do modelo;
- pipeline de inferência;
- formato das previsões.

Verificar que o pipeline de inferência preserva a quantidade e a ordem das observações.

---

# 26. GIT

## Regra crítica

Você está autorizado a utilizar Git localmente.

Você pode executar:

```bash
git status
git add
git commit
git log
git diff
git branch
```

Você pode criar commits locais durante o desenvolvimento.

### NÃO execute:

```bash
git push
```

sem autorização explícita do usuário.

Também não execute:

```bash
git pull
```

sem autorização explícita, pois mudanças externas devem ser coordenadas deliberadamente.

## Commits

Faça commits após milestones relevantes, e não após cada alteração trivial.

Exemplos:

```text
initial project structure
dataset audit and baseline
eda and preprocessing
baseline model suite
tree-based model experiments
gradient boosting experiments
feature engineering experiments
hyperparameter tuning
ensemble experiments
overfitting bootstrap analysis
final model
shap analysis
inference pipeline
documentation and reproducibility
```

Mensagens de commit devem ser claras, curtas e descritivas.

## Antes de qualquer commit

Verificar:

```bash
git status
git diff
```

Não commitar:

- senhas;
- API keys;
- tokens;
- credenciais;
- arquivos temporários;
- caches;
- ambientes virtuais;
- grandes artefatos desnecessários;
- informações sensíveis.

Garantir que `.gitignore` esteja configurado corretamente.

---

# 27. PUSH FINAL

O usuário controla o momento do push.

Nunca fazer push automaticamente.

Quando o usuário disser explicitamente:

> FAÇA O PUSH PARA O GITHUB

então:

1. verificar `git status`;
2. revisar alterações;
3. verificar ausência de secrets;
4. revisar commits;
5. confirmar branch atual;
6. somente então executar `git push`.

Se existir alguma dúvida sobre o remote ou branch, mostrar a informação ao usuário antes de alterar algo.

---

# 28. SEGURANÇA DO REPOSITÓRIO

Nunca expor:

- credenciais;
- tokens;
- API keys;
- dados privados;
- secrets.

Inspecionar `.gitignore`.

Nunca adicionar automaticamente `.env` ao Git.

---

# 29. QUALIDADE DE CÓDIGO

Código deve ser:

- legível;
- modular;
- tipado quando isso aumentar a clareza;
- documentado quando necessário;
- testável;
- reproduzível.

Evitar:

- código duplicado;
- variáveis sem significado;
- hardcoding desnecessário;
- notebooks que fazem tudo sem estrutura;
- funções gigantes;
- preprocessing duplicado.

Preferir funções pequenas e responsabilidades bem separadas.

---

# 30. PRINCÍPIO DE EXPERIMENTAÇÃO

Não fazer mudanças aleatórias.

Para cada experimento relevante:

1. formular hipótese;
2. executar experimento;
3. medir resultado;
4. comparar com baseline;
5. registrar;
6. decidir se vale prosseguir.

Não considerar melhoria de validação como definitiva quando o ganho for pequeno e estiver dentro da variabilidade esperada.

---

# 31. COMPUTAÇÃO

Se houver limitação de hardware, priorizar experimentos informativos.

Usar paralelização quando segura.

Evitar processos que consumam recursos de maneira irresponsável.

Não executar dezenas de buscas redundantes.

Antes de uma busca grande, avaliar se os resultados preliminares justificam o custo.

---

# 32. NÃO FABRICAR RESULTADOS

Nunca afirmar:

- "este é o melhor modelo";
- "este modelo generaliza melhor";
- "este ensemble é superior";
- "não existe overfitting";

sem evidência experimental suficiente.

Sempre executar a análise necessária.

Quando houver incerteza, declarar a incerteza.

---

# 33. CRITÉRIO DE DECISÃO DO MODELO FINAL

A escolha final deve considerar conjuntamente:

1. desempenho preditivo;
2. estabilidade;
3. overfitting;
4. complexidade;
5. capacidade de inferência;
6. reprodutibilidade;
7. compatibilidade com SHAP.

Não utilizar uma métrica única de forma isolada.

---

# 34. PRINCÍPIO DE COMPETIÇÃO

A solução deve ser desenvolvida com mentalidade de competição de ML, porém respeitando integralmente:

- o dataset disponível;
- a metodologia oficial;
- ausência de leakage;
- reprodutibilidade;
- integridade da avaliação.

Não tentar explorar falhas do sistema de avaliação.

O objetivo é obter excelente desempenho por meio de modelagem legítima e tecnicamente sólida.

---

# 35. CONDIÇÃO DE FINALIZAÇÃO

Não considere o projeto concluído apenas porque existe um modelo com bom RMSE.

Antes de finalizar, verificar:

### Dataset

- [ ] dataset auditado;
- [ ] target identificado;
- [ ] missing values analisados;
- [ ] duplicatas analisadas;
- [ ] leakage auditado.

### Modelagem

- [ ] baseline;
- [ ] múltiplas famílias de modelos;
- [ ] feature engineering investigado;
- [ ] tuning;
- [ ] robustez avaliada;
- [ ] ensemble investigado quando pertinente.

### Overfitting

- [ ] RMSE treino;
- [ ] RMSE validação;
- [ ] bootstrap treino;
- [ ] bootstrap validação;
- [ ] IC 95%;
- [ ] regra oficial aplicada.

### Reprodutibilidade

- [ ] estrutura organizada;
- [ ] README;
- [ ] requirements;
- [ ] modelo salvo;
- [ ] pipeline de inferência;
- [ ] execução sem intervenção manual.

### SHAP

- [ ] análise global;
- [ ] interpretação global;
- [ ] análise local;
- [ ] interpretação local.

### Git

- [ ] histórico organizado;
- [ ] sem secrets;
- [ ] commits significativos;
- [ ] nenhum push sem autorização.

---

# 36. ORDEM INICIAL DE EXECUÇÃO

Ao iniciar o trabalho neste repositório, siga esta ordem:

1. ler este CLAUDE.md;
2. inspecionar o repositório;
3. verificar o dataset;
4. compreender as características do dataset;
5. propor um plano experimental;
6. executar a auditoria inicial;
7. criar baseline;
8. iniciar experimentação sistemática.

Não começar criando imediatamente um modelo final sofisticado.

Primeiro compreender os dados.

Depois experimentar.

Depois otimizar.

Depois consolidar.

---

# 37. REGRA FINAL

Seu trabalho não é simplesmente escrever código.

Seu trabalho é conduzir um processo rigoroso de descoberta do melhor sistema de regressão possível dentro das regras do desafio.

Sempre priorize:

**evidência experimental > intuição**

**generalização > resultado acidental**

**reprodutibilidade > hacks**

**qualidade do pipeline > notebook descartável**

**melhoria mensurável > complexidade gratuita**