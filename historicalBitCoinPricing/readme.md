# Historical pricing notes
### this script will take roughly 1 million years to run
> it could be optimized but again lazy
> not to mention how often do you really need to pull every trade ever
### it will not check to see if it has been run before
> dont run it twice you will have dupes

- index.html is just the source code from [here](http://api.bitcoincharts.com/v1/csv/)
- you could easily just change the nokogiri open to pull right from the site
- buuut I was pinging their server plenty and decided to spare them
> to make sure this info is uptodate either copy the source doe from above into index.html or make the suggested change
> that being said this is for historical testing so it being up to date isnt the most important thing as other apis will be responsible for pulling current price
