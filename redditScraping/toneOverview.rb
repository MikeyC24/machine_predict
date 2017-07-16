class ToneOverview
  attr_reader :document_tone_hash, :sentences_tone_hash

  def initialize( tone_hash)
    @document_tone_hash = my_hash["document_tone"]
    @sentences_tone_hash = my_hash["sentences_tone"]
  end
  def document_tone
    document_tone.each do |el|

      puts el
    end
  end
end