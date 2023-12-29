-For our code to work, make sure the .csv file is in the same directory as the code

-Our project is weather forecast. The code takes models input, today date input and weather
parameters to predict the next day parameters (or "tomorrow" of the "today date" from input).

-The dataset is taken from Kaggle, it is the daily weather of New Delhi from 2013 to 2017. 

-The code does't need any parameters

-The nature of weather forecast require constant update to the data, and our data ends at 
24-04-2017, the "today" date shoudn't be pass 7 days after the end date. If the date is pass
that, it means that the dataset needs updating to remain accurate. The models can technically
predict beyond that, but the accuracy is no difference than guessing, which is bad. So for 
demonstrating purpose, the "today" input should be a date between 25-04-2017 and 01-05-2017
or should be left empty. If you leave it empty, the models will print predictsion for the 
day after 24-04-2017, which is 25-04-2017.

-TLDR: today date can be left empty if used for demonstration purpose. If you want to predict
other days, that day shouldn't be 7 days after 24/04/2017.

git clone https://github.com/phaniiiiiiii/project_AI
pip install -q -r requirements.txt
python main.py
