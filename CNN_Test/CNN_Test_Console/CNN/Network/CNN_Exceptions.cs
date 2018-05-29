using System;
using System.Runtime.Serialization;

namespace ConvNeuralNetwork
{
    class WrongLayerException : Exception
    {
        public WrongLayerException()
        {

        }

        public WrongLayerException(string message) : base(message)
        {
            Console.WriteLine(message);
        }

        public WrongLayerException(string message, Exception innerException) : base(message, innerException)
        {
        }

        protected WrongLayerException(SerializationInfo info, StreamingContext context) : base(info, context)
        {
        }
    }
    class UndefinedLayerException : Exception
    {
        public UndefinedLayerException()
        {
        }

        public UndefinedLayerException(string message) : base(message)
        {
            Console.WriteLine(message);
        }

        public UndefinedLayerException(string message, Exception innerException) : base(message, innerException)
        {
        }

        protected UndefinedLayerException(SerializationInfo info, StreamingContext context) : base(info, context)
        {
        }
    }
    class ConfigParserException : Exception
    {
        public ConfigParserException()
        {
        }

        public ConfigParserException(string message) : base(message)
        {
            Console.WriteLine(message);

        }

        public ConfigParserException(string message, Exception innerException) : base(message, innerException)
        {
        }

        protected ConfigParserException(SerializationInfo info, StreamingContext context) : base(info, context)
        {
        }
    }
}
