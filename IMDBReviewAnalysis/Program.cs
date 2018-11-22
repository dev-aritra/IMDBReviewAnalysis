using IMDBReviewAnalysis.DataStructures;
using IMDBReviewAnalysis.Utils;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using System;
using System.IO;

namespace IMDBReviewAnalysis
{
    class Program
    {
        private static string BaseDatasetsLocation = @"../../../Data";
        private static string TrainDataPath = $"{BaseDatasetsLocation}/train.csv";
        

        private static string BaseModelsPath = @"../../../MLModels";
        private static string ModelPath = $"{BaseModelsPath}/SentimentModel.zip";


        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed:1);
            //BuildTrainEvaluateAndSaveModel(mlContext);
            //Console.WriteLine("=============== End of training processh ===============");


            TestSinglePrediction(mlContext);

        }

        private static void BuildTrainEvaluateAndSaveModel(MLContext mlContext)
        {
            TextLoader textLoader = TextLoaderUtil.CreateTextLoader(mlContext);
            IDataView trainingDataView = textLoader.Read(TrainDataPath);

            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText("Review", "Features");

            var trainer = mlContext.BinaryClassification.Trainers.FastTree(label: "Label", features: "Features");
            var trainingPipeLine = dataProcessPipeline.Append(trainer);

            Console.WriteLine("Training model");
            ITransformer trainModel = trainingPipeLine.Fit(trainingDataView);

            using (var fs = new FileStream(ModelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(trainModel, fs);
        }

        private static void TestSinglePrediction(MLContext mLContext)
        {
            while(Console.ReadLine()!= "quit" )
            {
                Console.WriteLine("Enter review");
                var text = Console.ReadLine();
                IMDBReview sample = new IMDBReview { Review = text };


                ITransformer trainedModel;
                using (var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                {
                    trainedModel = mLContext.Model.Load(stream);
                }

                var predFunction = trainedModel.MakePredictionFunction<IMDBReview, PredictionModel>(mLContext);
                var resultprediction = predFunction.Predict(sample);

                Console.WriteLine($"=============== Single Prediction  ===============");
                Console.WriteLine($"Text: {sample.Review} | Prediction: { (resultprediction.Prediction ? "Good review" : "Bad review")} | Probability: { resultprediction.Probability}");
                Console.WriteLine($"==================================================");
            }
            

        }
    }
}
