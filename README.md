Training The Machine End To end

Overview

The goal of this project is to implement an end-to-end data science process utilizing machine learning in ensemble.
Specifically to build a Flask application that takes in a user input (a stock market ticker) via an HTML request routed to an API call that returns a time series of stock market data for the stock associated with the ticker.

Also returned are additional features commonly used in market analysis, some of which may require that their explicit signals be computed and appended to the data. Also appended is a target for scores to be computed upon; a boolean indicating whether or not a particular stock would or would not be lower in a week.

An, of course, expandable dictionary of models and their parameters is then looped through and used to feed a GridSearchCV object. The scores of which are then returned as a table on an HTML page.



Data

Several endpoints of an API providing stock market data from AlphaVantage are called for each ticker a user may submit. One returns the daily market data associated with that ticker while the rest contain common stock market analysis features. These tables are then concatenated and the resulting table trimmed since some of the joined features do not span the entire range of dates in the market data table.

ML Process
After removing NaNs and collating the data I then split it into test and train splits and again into feature matrices and target arrays. These were then fit to a CrossGridCV model as it iterated through a dictionary of ML models and their parameters.

Model Chosen, Why, Evaluated How?
Since this was being treated as a classification problem, and to further explore ensembling I chose to focus on a random forest model. Evaluating the best model I could obtain through a range of models tested and returning that model's scores to the user.

Flask Application
The lecture and lesson the following week were useful starting points to understanding Flask conceptually and then implementing after some further research and documentation reading. I chose to utilize jinja to reach some degree of modularity with my HTML page and routings. A small amount of CSS is included to aid in the readability of the ML results page.

Result
The application can indeed accept user input and use it to conduct a sequence of steps from the data pipeline to an HTML table display of ML scores.