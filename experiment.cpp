// Read file

#include <iostream.h>
#include <fstream.h>
#include <string.h>

using namespace std;
typedef vector<Vector3>	coordVec;

void readCoords(coordVec * coords, const char * fileName, const double multiply=1)
{
	double x,y,z;
	bool x_eq, y_eq, z_eq;
	bool mantiss = false;
	ifstream fin(fileName);
	if (fin)
	{
		x_eq = true;
		y_eq = z_eq = false;
		x = y = z = 0;
		double mant = 0;
		double mul = 10;
		char ch;
		int sign = 1;
		bool change = true;
		mantiss = false;
		while(fin.get(ch))
		{
			if (change)
				if (ch == '-')
				{
					sign = -1;
					change = false;
					continue;
				}
				else
					sign = 1;

			if (ch == '.')
			{
				mantiss = true;
				mant = 0;
				change = false;
				continue;
			}
			if (ch < '0' || ch > '9')
			{
				if (x_eq)
				 	x += mant;
				else if (y_eq)
				 	y += mant;
				else
				{
				 	z += mant;
				 	coords->push_back(Vector3(x*multiply,y*multiply,z*multiply));
				 	x = y = z = 0;
				}
				bool old_x = x_eq;
				x_eq = z_eq;
				z_eq = y_eq;
				y_eq = old_x;
				mant = 0;
				mul = 10;
				mantiss = false;
				change = true;
				continue;
			}
			change = false;
			if (!mantiss)
			{
				if (x_eq)
					x = x*10 + sign*(int)(ch-'0');
				else if (y_eq)
					y = y*10 + sign*(int)(ch-'0');
				else if (z_eq)
					z = z*10 + sign*(int)(ch-'0');
			}
			else {
				mant += (int)(ch-'0')/mul;
				mul *= 10;
			}
		}
	}

	fin.close();
}

int main()
{
	coordVec * coords = new coordVec;
	readCoords(coords, "coords.txt");

	for (coordVec::size_type i=0; i < coords->size(); i++)
	    	{
	    		cout << (*coords)[i] << endl;
	    	}

	return 0;
}
