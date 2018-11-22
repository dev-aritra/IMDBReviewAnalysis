using Microsoft.ML;
using Microsoft.ML.Runtime.Data;

namespace IMDBReviewAnalysis.Utils
{
    public static class TextLoaderUtil
    {
        public static TextLoader CreateTextLoader(MLContext mLContext)
        {
            TextLoader textLoader = mLContext.Data.TextReader(new TextLoader.Arguments()
            {
                Separator=",",
                HasHeader=true,
                Column= new[]
                {
                    new TextLoader.Column("Review", DataKind.Text,0),
                    new TextLoader.Column("Label", DataKind.Bool, 1)
                }

            });
            return textLoader;

        }
    }
}
