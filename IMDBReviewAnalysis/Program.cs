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
        private static string basedataSetLoc = @"../../../Data";
        private static string trainingSetLoc = $"{basedataSetLoc}/training_data.csv";


        private static string baseModelLoc = @"../../../MLModels";
        private static string generatedModelLoc = $"{baseModelLoc}/SentimentModel.zip";


        static void Main(string[] args)
        {
            Console.ForegroundColor = ConsoleColor.Blue;

            MLContext mlContext = new MLContext(seed: 1);
            if (GetOperationToBePerformedFromUser())
            {
                Console.WriteLine("Training model in process");
                BuildTrainEvaluateAndSaveModel(mlContext);
                Console.WriteLine("Training complete");
                if (GetOperationToBePerformedFromUser(true))
                {
                    TestSinglePrediction(mlContext);
                }
            }
            else
            {
                TestSinglePrediction(mlContext);
            }
            
            

        }

        private static bool GetOperationToBePerformedFromUser()
        {
            Console.WriteLine("What do you want to do?");
            Console.WriteLine("1. Train model");
            Console.WriteLine("2. Test model");
            Console.WriteLine("=====================================================================================================");

            var input = Console.ReadLine();
            if (input.Equals("1"))
            {
                return true;
            }
            else if (input.Equals("2"))
            {
                return false;
            }
            else
            {
                Console.WriteLine("Invalid input");
                return GetOperationToBePerformedFromUser();
            }
        }

        private static bool GetOperationToBePerformedFromUser(bool shouldTest)
        {

            Console.WriteLine("Do you want to test the model now? Y/N");
            var input = Console.ReadLine();
            if (input.Equals("Y") || input.Equals("y"))
            {
                return true;
            }
            return false;

        }

        private static void BuildTrainEvaluateAndSaveModel(MLContext mlContext)
        {
            TextLoader textLoader = TextLoaderUtil.CreateTextLoader(mlContext);
            IDataView trainingDataView = textLoader.Read(trainingSetLoc);

            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText("Review", "Features");

            var trainer = mlContext.BinaryClassification.Trainers.FastTree(label: "Label", features: "Features");
            

            var trainingPipeLine = dataProcessPipeline.Append(trainer);
            ITransformer trainModel = trainingPipeLine.Fit(trainingDataView);

            using (var fs = new FileStream(generatedModelLoc, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(trainModel, fs);
        }

        private static void TestSinglePrediction(MLContext mLContext)
        {
            while (true)
            {
                Console.WriteLine();
                Console.WriteLine("Enter review or type quit");
                var text = Console.ReadLine();
                if(text.ToLower().Equals("quit"))
                {
                    break;
                }
                IMDBReview sample = new IMDBReview { Review = text };


                ITransformer trainedModel;
                using (var stream = new FileStream(generatedModelLoc, FileMode.Open, FileAccess.Read, FileShare.Read))
                {
                    trainedModel = mLContext.Model.Load(stream);
                }

                var predFunction = trainedModel.MakePredictionFunction<IMDBReview, PredictionModel>(mLContext);
                var resultprediction = predFunction.Predict(sample);

                Console.WriteLine("=====================================================================================================");

                Console.ForegroundColor = resultprediction.Prediction ? ConsoleColor.Red : ConsoleColor.Green;

                Console.WriteLine($"Text: {sample.Review} | Prediction: { (resultprediction.Prediction ? "Negetive review" : "Positive review")}");
                Console.ForegroundColor = ConsoleColor.Blue;
                Console.WriteLine("=====================================================================================================");
            }


        }
    }
}
