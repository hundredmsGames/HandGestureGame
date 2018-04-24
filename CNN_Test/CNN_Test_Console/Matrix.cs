using System;

namespace MatrixLib
{
	public class Matrix
	{
        #region Variables

        public int rows;
		public int cols;
		public double[,] data;

		static Random randomize = new Random();

        #endregion

        #region Constructors

        public Matrix (int rows, int cols)
		{
			this.rows = rows;
			this.cols = cols;

			data = new double[rows, cols];
		}

		// Copy Constructor
		public Matrix(Matrix m)
		{
			this.rows = m.rows;
			this.cols = m.cols;

			this.data = new double[rows, cols];

			for(int i = 0; i < rows; i++)
				for(int j = 0; j < cols; j++)
					this.data[i, j] = m.data[i, j];
		}

        public double this[int r, int c]
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

        public static Matrix Add(Matrix m, double x)
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

        public static Matrix Subtract(Matrix m, double x)
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

        public static Matrix Multiply(Matrix m, double x)
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

        public static Matrix Divide(Matrix m, double x)
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

        public static Matrix Divide(double x, Matrix m)
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

        #region Dot Product

        public static double DotProduct(Matrix m1, Matrix m2, int m1_r, int m1_c)
        {
            double sum = 0.0f;

            for(int i = 0; i < m2.rows; i++)
            {
                for(int j = 0; j < m2.cols; j++)
                {
                    sum += m2[i, j] * m1[m1_r + i, m1_c + j];
                }
            }

            return sum;
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
            return Multiply(m1, m2);
        }

        public static Matrix operator /(Matrix m1, Matrix m2)
        {
            return Divide(m1, m2);
        }

        public static Matrix operator +(Matrix m, double x)
        {
            return Add(m, x);
        }

        public static Matrix operator +(double x, Matrix m)
        {
            return Add(m, x);
        }

        public static Matrix operator -(Matrix m, double x)
        {
            return Subtract(m, x);
        }

        public static Matrix operator -(double x, Matrix m)
        {
            return Negative(Subtract(m, x));
        }

        public static Matrix operator *(Matrix m, double x)
        {
            return Multiply(m, x);
        }

        public static Matrix operator *(double x, Matrix m)
        {
            return Multiply(m, x);
        }

        public static Matrix operator /(Matrix m, double x)
        {
            return Divide(m, x);
        }

        public static Matrix operator /(double x, Matrix m)
        {
            return Divide(x, m);
        }

        public static Matrix operator -(Matrix m)
        {
            return Negative(m);
        }

        #endregion

        #region Map Functions

        // Maps the matrix according to given func.
        // i.e: Changes each element according to given func.
        public void Map(Func<double, double> mapFunc)
		{
			for(int i = 0; i < this.rows; i++)
			{
				for(int j = 0; j < this.cols; j++)
				{
					this.data[i, j] = mapFunc(this.data[i, j]);
				}
			}
		}

		// Same as the non-static version of Map
		public static Matrix Map(Matrix m, Func<double, double> mapFunc)
		{
			Matrix mapped = new Matrix(m.rows, m.cols);
			for(int i = 0; i < m.rows; i++)
			{
				for(int j = 0; j < m.cols; j++)
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
			for(int i = 0; i < rows; i++)
			{
				for(int j = 0; j < cols; j++)
				{
					this.data[i, j] = randomize.NextDouble() * 2.0 - 1.0;
				}
			}
		}

        #endregion

        #region Transform Methods
        // Convert matrix from array.
        public static Matrix FromArray(double[] arr)
		{
			Matrix m = new Matrix(arr.Length, 1);

			for(int i = 0; i < arr.Length; i++)
				m.data[i, 0] = arr[i];

			return m;
		}

		// Convert matrix to array.
		public static double[] ToArray(Matrix m)
		{
			double[] arr = new double[m.rows * m.cols];

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
					ret += data[i, j].ToString("F4") + "\t";
				}
				ret += "\n";
			}

			return ret;
		}

        #endregion
    }
}

