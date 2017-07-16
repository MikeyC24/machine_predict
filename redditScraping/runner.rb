require_relative './helper'
require 'redd'
 

 # log in details will have to come out of here if we plan on using non testing details as git hub is public  

login = {
	reddit: {
		email: "coinscrapebot@gmail.com",
		username: "coinscrapebottest1",
		password: "Fuckfuck1",
		secret: "jJJI0rL5pt8E_fJFoJZfRNzA6Oc",
		client_id: "0cBhORRt58zpzg"

	},
	gmail: {
		email: "coinscrapebot@gmail.com",
		password: "Fuckfuck1"
	}
}


session = Redd.it(
  user_agent: 'Redd:RandomBot:v1.0.0 (by /u/Mustermind)',
  client_id:  login[:reddit][:client_id],
  secret:     login[:reddit][:secret],
  username:   login[:reddit][:username],
  password:   login[:reddit][:password]
)

session.subreddit('Bitcoin').search.comments(" ", {limit: 1})

# im working from windows so naturally noting works... I will have to take another crack at this later 