require 'open-uri'
require 'json'
require 'sqlite3'
require 'date'

database = File.new("poloniex24hourvolume.db", "a")
database.close
db = SQLite3::Database.new("poloniex24hourvolume.db")

def timestamp
  Time.now.to_i
end

sql_command = <<-SQL
CREATE TABLE IF NOT EXISTS trades(
  insertedTime string NOT NULL,
  currency1 string NOT NULL,
  currency1Volume string NOT NULL,
  currency2 string NOT NULL,
  currency2Volume string NOT NULL);
SQL
db.execute(sql_command)

trades = JSON.parse(open('https://poloniex.com/public?command=return24hVolume').read)

sql_command = <<-SQL
SELECT insertedTime FROM trades ORDER BY insertedTime DESC LIMIT 1;
SQL
temp = db.execute(sql_command)

if Time.at(temp[0][0]).to_date != Time.at(timestamp).to_date

		trades.each do |i|
			if i[1].class == Hash
					temp = []
				i[1].each do |trade|
					temp << trade
				end
				temp.flatten!

				sql_command = <<-SQL
				INSERT INTO trades(
				 insertedTime,
				 currency1,
				 currency1Volume,
				 currency2,
				 currency2Volume
			 )
				VALUES
				 (
				 "#{timestamp}",
				 "#{temp[0]}",
				 "#{temp[1]}",
				 "#{temp[2]}",
				 "#{temp[3]}");
				 SQL
				 db.execute(sql_command)
			end
		end
		p "This script inserted #{trades.length} trades into the database"
	else
		p "This script already ran today"
end

# sleep(1.days)
