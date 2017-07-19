require 'open-uri'
require 'mongo'

res = open('https://poloniex.com/public?command=return24hVolume')

mongodb = {
	username: "temp",
	password: "temp",
	db_name: "temp"
}

uri = "mongodb://#{mongodb[:username]}:#{mongodb[:password]}@#{mongodb[:db_name]}"

client = Mongo::Client.new(uri)

coin_info = client[:coin_info]

coin_info.insert_many(coin_info)
