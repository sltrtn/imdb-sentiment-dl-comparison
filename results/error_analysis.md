# DistilBERT error analysis

Checkpoint: `models/distilbert-lr2e-05` — test accuracy **0.9141** (22,853/25,000)

## Accuracy by review length

| Words | Reviews | Accuracy |
|---|---|---|
| 4–124 | 6,214 | 0.9340 |
| 125–169 | 6,192 | 0.9381 |
| 170–273 | 6,320 | 0.9201 |
| 274–2235 | 6,274 | 0.8647 |

## Negation slice

| Slice | Reviews | Accuracy | Error rate |
|---|---|---|---|
| with negation cue | 18,743 | 0.9081 | 0.0919 |
| without negation cue | 6,257 | 0.9322 | 0.0678 |

## Most confident false positives

- **confidence 0.998**, 79 words — “This movie was pure genius. John Waters is brilliant. It is hilarious and I am not sick of it even after seeing it about 20 times since I bought it a few months ago. The acting is great, although Ricki Lake could have been better. And Johnny Depp is magnificent. He is such a beautiful man and a very talented actor. And seeing most of John …”
- **confidence 0.998**, 302 words — “Lois Weber's film "Hypocrites" was and still kind of is a very bold and daring film. I enjoyed it and was very impressed by the filming and story of it. The priest sees the hypocrisy of the people in his church and tries to show them the "naked" truth. The people are appalled when he reveals the naked statue portraying truth, after failin …”
- **confidence 0.997**, 259 words — “I went to see this 3 nights ago here in Cork, Ireland. It was the world premiere of it, in the tiny cinema in the Triskel Arts Centre as part of the Cork Film Festival. I found "Strange Fruit" to be an excellent movie. It is a bit rough around the edges, but for a low-budget movie that is to be expected! In general the acting (particularl …”
- **confidence 0.997**, 45 words — “I really liked this quirky movie. The characters are not the bland beautiful people that show up in so many movies and on TV. It has a realistic edge, with a captivating story line. The main title sequence alone makes this movie fun to watch.”
- **confidence 0.997**, 197 words — “This film derives from a Long Running ITV sitcom by the same name.The Sitcom lasted for half a decade roughly and brought to our screens Rigsby,Phillip,Alan,Mrs Jones & Vienna. Then in 1980 The film version hit the Cinemas.Now when it did,sadly Richard Beckinsale had passed away & was replaced by Only when i laugh actor Chris Strauli. I m …”

## Most confident false negatives

- **confidence 0.997**, 674 words — “Forget everything that you have ever read about the Mallachi Brothers' straight-to-video release "Snakes on a Train," especially if it was a negative review. This movie is way more fun than the movie that it obviously rips off: "Snakes on a Plane." Frankly, I am surprised that more people aren't rhapsodizing about this low-budget Asylum R …”
- **confidence 0.997**, 347 words — “**SPOILERS AHEAD** It is really unfortunate that a movie so well produced turns out to be such a disappointment. I thought this was full of (silly) clichés and that it basically tried to hard. To the (American) guys out there: how many of you spend your time jumping on your girlfriend's bed and making monkey sounds? To the (married) girls …”
- **confidence 0.997**, 136 words — “We just saw this film previewed before release at the Norfolk (VA) Film Forum, and there was general agreement on two matters: There were excellent performances in a first rate drama by the two leads and by others: and secondly, the marketing for this movie will only bring disaster. We saw a lurid poster with chains and suggestive comment …”
- **confidence 0.997**, 304 words — “Yes, indeed we have a winner- a winner in best dumb-action-movie! The only reason I chose to vote a 10 for this movie is because it's so incredibly bad-made that it actually becomes funny. "Night Hunter" is basically about Jack Cutter, a Vampire hunter (Vampire hunters have been in his generation for centuries, apparently), and his missio …”
- **confidence 0.996**, 190 words — “I am a big follower of Indian Movies especially Malayalam and Tamil. Shame on India for not sending this movie as their official Oscar entry. I have seen this movie and it has clearly revealed to me the maturity Tamil cinema has in its screenplay and narrative which bollywood better catch up with.By the way to all we westerners, Tamil Cin …”
