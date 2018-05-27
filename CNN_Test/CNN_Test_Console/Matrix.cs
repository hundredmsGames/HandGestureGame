using System;

namespace MatrixLib
{
	public class Matrix
	{
        #region Variables

        public int rows;
		public int cols;
		public float[,] data;

		static Random randomize = new Random();

        #endregion

        #region Constructors
        //this one is for deserializing objects
        public Matrix()
        {

        }

        public Matrix(int rows, int cols)
		{
			this.rows = rows;
			this.cols = cols;

			data = new float[this.rows, this.cols];
		}

        public Matrix(float[] arr)
        {
            this.rows = arr.Length;
            this.cols = 1;
            this.data = new float[this.rows, this.cols];

            for (int i = 0; i < arr.Length; i++)
                data[i, 0] = arr[i];
        }

        public Matrix(byte[][] pixels)
        {
            this.rows = pixels.Length;
            this.cols = pixels.Length;
            this.data = new float[this.rows, this.cols];

            for (int i = 0; i < pixels.Length; i++)
                for (int j = 0; j < pixels.Length; j++)
                    data[i, j] = pixels[i][j];
        }

        // Copy Constructor
        public Matrix(Matrix m)
		{
			this.rows = m.rows;
			this.cols = m.cols;

			this.data = new float[rows, cols];

			for(int i = 0; i < rows; i++)
				for(int j = 0; j < cols; j++)
					this.data[i, j] = m.data[i, j];
		}

        public float this[int r, int c]
        {
            get
            {
                return data[r, c];
            }

            set
            {
                data[r, c] = value;
            }
        }

        #endregion

        #region Element-wise Operations

        public static Matrix Add(Matrix m1, Matrix m2)
        {
            if (m1.rows != m2.rows || m1.cols != m2.cols)
            {
                Console.WriteLine("Error: rows and cols should be same!");
                return null;
            }

            Matrix sum = new Matrix(m1.rows, m1.cols);
            for (int i = 0; i < sum.rows; i++)
            {
                for (int j = 0; j < sum.cols; j++)
                {
                    sum.data[i, j] = m1.data[i, j] + m2.data[i, j];
                }
            }

            return sum;
        }

        public static Matrix Subtract(Matrix m1, Matrix m2)
        {
            if (m1.rows != m2.rows || m1.cols != m2.cols)
            {
                Console.WriteLine("Error: rows and cols should be same!");
                return null;
            }

            Matrix sub = new Matrix(m1.rows, m1.cols);
            for (int i = 0; i < sub.rows; i++)
            {
                for (int j = 0; j < sub.cols; j++)
                {
                    sub.data[i, j] = m1.data[i, j] - m2.data[i, j];
                }
            }

            return sub;
        }

        // Hadamard Multiply
        public static Matrix Multiply(Matrix m1, Matrix m2)
        {
            if (m1.rows != m2.rows || m1.cols != m2.cols)
            {
                Console.WriteLine("Error: rows and cols should be same!");
                return null;
            }

            Matrix mult = new Matrix(m1.rows, m1.cols);
            for (int i = 0; i < mult.rows; i++)
            {
                for (int j = 0; j < mult.cols; j++)
                {
                    mult.data[i, j] = m1.data[i, j] * m2.data[i, j];
                }
            }

            return mult;
        }

        public static Matrix Divide(Matrix m1, Matrix m2)
        {
            if (m1.rows != m2.rows || m1.cols != m2.cols)
            {
                Console.WriteLine("Error: rows and cols should be same!");
                return null;
            }

            Matrix div = new Matrix(m1.rows, m1.cols);
            for (int i = 0; i < div.rows; i++)
            {
                for (int j = 0; j < div.cols; j++)
                {
                    if (m2.data[i, j] == 0.0)
                    {
                        Console.WriteLine("Error: Cannot divide by zero!");
                        return null;
                    }

                    div.data[i, j] = m1.data[i, j] / m2.data[i, j];
                }
            }

            return div;
        }

        #endregion

        #region Scalar Operations

        public static Matrix Add(Matrix m, float x)
        {
            Matrix sum = new Matrix(m.rows, m.cols);
            for (int i = 0; i < sum.rows; i++)
            {
                for (int j = 0; j < sum.cols; j++)
                {
                    sum.data[i, j] = m.data[i, j] + x;
                }
            }

            return sum;
        }

        public static Matrix Subtract(Matrix m, float x)
        {
            Matrix sub = new Matrix(m.rows, m.cols);
            for (int i = 0; i < sub.rows; i++)
            {
                for (int j = 0; j < sub.cols; j++)
                {
                    sub.data[i, j] = m.data[i, j] - x;
                }
            }

            return sub;
        }

        public static Matrix Multiply(Matrix m, float x)
        {
            Matrix mult = new Matrix(m.rows, m.cols);
            for (int i = 0; i < mult.rows; i++)
            {
                for (int j = 0; j < mult.cols; j++)
                {
                    mult.data[i, j] = m.data[i, j] * x;
                }
            }

            return mult;
        }

        public static Matrix Divide(Matrix m, float x)
        {
            Matrix div = new Matrix(m.rows, m.cols);
            for (int i = 0; i < div.rows; i++)
            {
                for (int j = 0; j < div.cols; j++)
                {
                    div.data[i, j] = m.data[i, j] / x;
                }
            }

            return div;
        }

        public static Matrix Divide(float x, Matrix m)
        {
            Matrix div = new Matrix(m.rows, m.cols);
            for (int i = 0; i < div.rows; i++)
            {
                for (int j = 0; j < div.cols; j++)
                {
                    div.data[i, j] = x / m.data[i, j];
                }
            }

            return div;
        }

        public static Matrix Negative(Matrix m)
        {
            Matrix neg = new Matrix(m.rows, m.cols);
            for (int i = 0; i < neg.rows; i++)
            {
                for (int j = 0; j < neg.cols; j++)
                {
                    neg.data[i, j] = -m.data[i, j];
                }
            }

            return neg;
        }

        public void FillZero()
        {
            for (int i = 0; i < this.rows; i++)
                for (int j = 0; j < this.cols; j++)
                    this[i, j] = 0f;
        }

        #endregion

        #region Matrix Multiply (Matrix Product)

        public static Matrix SlowMultiply(Matrix m1, Matrix m2)
		{
			if(m1.cols != m2.rows)
			{
				Console.WriteLine("Error: Cannot get product of these two matrixes, sizes does not match!");
				return null;
			}

			Matrix product = new Matrix(m1.rows, m2.cols);
			for(int i = 0; i < m1.rows; i++)
			{
				for(int j = 0; j < m2.cols; j++)
				{
					for(int k = 0; k < m1.cols; k++)
					{
						product.data[i, j] += m1.data[i, k] * m2.data[k, j]; 
					}
				}
			}

			return product;
		}

        #endregion

        #region Transpose

        // Transpose of matrix m.
        public static Matrix Transpose(Matrix m)
		{
			Matrix transpose = new Matrix(m.cols, m.rows);

			for(int i = 0; i < m.rows; i++)
			{
				for(int j = 0; j < m.cols; j++)
				{
					transpose.data[j, i] = m.data[i, j];
				}
			}

			return transpose;
		}

        #endregion

        #region Normalization

        public void Normalize(float oldMin, float oldMax, float newMin, float newMax)
        {
            for (int i = 0; i < this.rows; i++)
            {
                for (int j = 0; j < this.cols; j++)
                {
                    this[i, j] = ((this[i, j] - oldMin) / (oldMax - oldMin)) * (newMax - newMin) + newMin;
                }
            }
        }

        public static Matrix Normalize(Matrix matrix, float oldMin, float oldMax, float newMin, float newMax)
        {
            Matrix normalized = new Matrix(matrix.rows, matrix.cols);

            for (int i = 0; i < matrix.rows; i++)
            {
                for (int j = 0; j < matrix.cols; j++)
                {
                    normalized[i, j] = ((matrix[i, j] - oldMin) / (oldMax - oldMin)) * (newMax - newMin) + newMin;
                }
            }

            return normalized;
        }

        #endregion

        #region Map Functions

        // Maps the matrix according to given func.
        // i.e: Changes each element according to given func.
        public void Map(Func<float, float> mapFunc)
        {
            for (int i = 0; i < this.rows; i++)
            {
                for (int j = 0; j < this.cols; j++)
                {
                    this.data[i, j] = mapFunc(this.data[i, j]);
                }
            }
        }

        // Same as the non-static version of Map
        public static Matrix Map(Matrix m, Func<float, float> mapFunc)
        {
            Matrix mapped = new Matrix(m.rows, m.cols);
            for (int i = 0; i < m.rows; i++)
            {
                for (int j = 0; j < m.cols; j++)
                {
                    mapped.data[i, j] = mapFunc(m.data[i, j]);
                }
            }

            return mapped;
        }

        #endregion

        #region Randomize Method

        // Randomize numbers in matrix
        public void Randomize()
        {
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    this.data[i, j] = (float) (randomize.NextDouble() * 2.0 - 1.0);
                }
            }
        }

        #endregion

        #region Get Max Row Index

        public int GetMaxRowIndex()
        {
            int maxRow = 0;
            for (int i = 0; i < this.rows; i++)
            {
                if (this[i, 0] > this[maxRow, 0])
                    maxRow = i;
            }

            return maxRow;
        }

        #endregion

        #region Operator Overloading

        public static Matrix operator +(Matrix m1, Matrix m2)
        {
            return Add(m1, m2);
        }

        public static Matrix operator -(Matrix m1, Matrix m2)
        {
            return Subtract(m1, m2);
        }

        public static Matrix operator *(Matrix m1, Matrix m2)
        {
            return SlowMultiply(m1, m2);
        }

        public static Matrix operator /(Matrix m1, Matrix m2)
        {
            return Divide(m1, m2);
        }

        public static Matrix operator +(Matrix m, float x)
        {
            return Add(m, x);
        }

        public static Matrix operator +(float x, Matrix m)
        {
            return Add(m, x);
        }

        public static Matrix operator -(Matrix m, float x)
        {
            return Subtract(m, x);
        }

        public static Matrix operator -(float x, Matrix m)
        {
            return Negative(Subtract(m, x));
        }

        public static Matrix operator *(Matrix m, float x)
        {
            return Multiply(m, x);
        }

        public static Matrix operator *(float x, Matrix m)
        {
            return Multiply(m, x);
        }

        public static Matrix operator /(Matrix m, float x)
        {
            return Divide(m, x);
        }

        public static Matrix operator /(float x, Matrix m)
        {
            return Divide(x, m);
        }

        public static Matrix operator -(Matrix m)
        {
            return Negative(m);
        }

        #endregion

        #region Transform Methods

        public static Matrix IncreaseToTwoDimension(Matrix m, int rows, int cols)
        {
            Matrix increased = new Matrix(rows, cols);
            int idx = 0;

            for(int i = 0; i < rows; i++)
            {
                for(int j = 0; j < cols; j++)
                {
                    increased[i, j] = m[idx++, 0];
                }
            }

            return increased;
        }

        public static Matrix DecreaseToOneDimension(Matrix m)
        {
            Matrix reduced = new Matrix(m.rows * m.cols, 1);
            int idx = 0;

            for (int i = 0; i < m.rows; i++)
            {
                for (int j = 0; j < m.cols; j++)
                {
                    reduced[idx++, 0] = m[i, j];
                }
            }

            return reduced;
        }

        // Convert matrix from array.
        public static Matrix FromArray(float[] arr)
		{
			Matrix m = new Matrix(arr.Length, 1);

			for(int i = 0; i < arr.Length; i++)
				m.data[i, 0] = arr[i];

			return m;
		}

		// Convert matrix to array.
		public static float[] ToArray(Matrix m)
		{
            float[] arr = new float[m.rows * m.cols];

			int cnt = 0;
			for(int i = 0; i < m.rows; i++)
			{
				for(int j = 0; j < m.cols; j++)
				{
					arr[cnt++] = m.data[i, j];
				}
			}

			return arr;
		}

        #endregion

        #region ToString Override

        public override string ToString()
		{
			string ret = "";
			ret += string.Format("rows: {0}, cols: {1}\n", rows, cols);

			for(int i = 0; i < rows; i++)
			{
				for(int j = 0; j < cols; j++)
				{
					ret += data[i, j].ToString("F6") + "\t";
				}
				ret += "\n";
			}

			return ret;
		}

        #endregion
    }
}

