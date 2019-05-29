using System;
using System.IO;
using Microsoft.ML;
using SentimentAnalysisConsoleApp.DataStructures;
using static Microsoft.ML.DataOperationsCatalog;

namespace AnaliseSentimento
{
    class Program
    {


        private static readonly string BaseDatasetsRelativePath = "../../../Data/";
        private static readonly string DataRelativePath = $"{BaseDatasetsRelativePath}wikiDetoxAnnotated40kRows.tsv";

        private static readonly string DataPath = GetAbsolutePath(DataRelativePath);

        private static readonly string BaseModelsRelativePath = "../../../MLModels/";
        private static readonly string ModelRelativePath = $"{BaseModelsRelativePath}SentimentModel.zip";

        private static readonly string ModelPath = GetAbsolutePath(ModelRelativePath);

        static void Main(string[] args)
        {

            // ML NET
            var mlContext = new MLContext(seed: 1);

            // Tipos de Dados
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(DataPath, hasHeader: true);

            TrainTestData trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            IDataView trainingData = trainTestSplit.TrainSet;

            // PipeLine
            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentIssue.Text));


            // Algoritmo que será aplicado.
            var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // Treinar
            ITransformer trainedModel = trainingPipeline.Fit(trainingData);


            Console.WriteLine("The model is saved to {0}", ModelPath);

            // Frases para treinar
            SentimentIssue  sampleStatement = new SentimentIssue { Text = "This is a very shit movie" };
            SentimentIssue sampleStatement1 = new SentimentIssue { Text = "This is a very good movie" };
            SentimentIssue sampleStatement2 = new SentimentIssue { Text = "awesome movie" };


            // Predizer as frases
            var predEngine = mlContext.Model.CreatePredictionEngine<SentimentIssue, SentimentPrediction>(trainedModel);

            // Score
            var resultprediction = predEngine.Predict(sampleStatement);
            var resultprediction1 = predEngine.Predict(sampleStatement1);
            var resultprediction2 = predEngine.Predict(sampleStatement2);


            Console.WriteLine($"=============== Single Prediction  ===============");
            Console.WriteLine($"Text:  { sampleStatement.Text } | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "Toxic" : "Non Toxic")} sentiment | Probability of being toxic: {resultprediction.Probability} ");
            Console.WriteLine($"Text: { sampleStatement1.Text } | Prediction: {(Convert.ToBoolean(resultprediction1.Prediction) ? "Toxic" : "Non Toxic")} sentiment | Probability of being toxic: {resultprediction1.Probability} ");
            Console.WriteLine($"Text: { sampleStatement2.Text } | Prediction: {(Convert.ToBoolean(resultprediction2.Prediction) ? "Toxic" : "Non Toxic")} sentiment | Probability of being toxic: {resultprediction2.Probability} ");
            Console.WriteLine($"================End of Process.Hit any key to exit==================================");

        }

         public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath , relativePath);

            return fullPath;
        }
    }
}
