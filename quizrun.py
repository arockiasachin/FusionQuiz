import re
import operator
from keybert import KeyBERT
def is_number(s):
    try:
        float(s) if '.' in s else int(s)
        return True
    except ValueError:
        return False

def load_stop_words(stop_word_file):
    stop_words = []
    with open(stop_word_file, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip()[0:1] != "#":
                for word in line.split():
                    stop_words.append(word)
    return stop_words

def build_stop_word_regex(stop_word_file_path):
    stop_word_list = load_stop_words(stop_word_file_path)
    stop_word_regex_list = [r'\b' + re.escape(word) + r'(?![\w-])' for word in stop_word_list]
    stop_word_pattern = re.compile('|'.join(stop_word_regex_list), re.IGNORECASE)
    return stop_word_pattern

def separate_words(text, min_word_return_size):
    splitter = re.compile('[^a-zA-Z0-9_\\+\\-/]')
    words = []
    for single_word in splitter.split(text):
        current_word = single_word.strip().lower()
        if len(current_word) > min_word_return_size and not is_number(current_word):
            words.append(current_word)
    return words

def split_sentences(text):
    sentence_delimiters = re.compile('[.!?,;:\t\\\\"\\(\\)\\\'\u2019\u2013]|\\s\\-\\s')
    sentences = sentence_delimiters.split(text)
    return sentences

def generate_candidate_keywords(sentence_list, stopword_pattern):
    phrase_list = []
    for s in sentence_list:
        tmp = re.sub(stopword_pattern, '|', s.strip())
        phrases = tmp.split("|")
        for phrase in phrases:
            phrase = phrase.strip().lower()
            if phrase:
                phrase_list.append(phrase)
    return phrase_list

def calculate_word_scores(phrase_list):
    word_frequency = {}
    word_degree = {}
    for phrase in phrase_list:
        word_list = separate_words(phrase, 0)
        word_list_length = len(word_list)
        word_list_degree = word_list_length - 1
        for word in word_list:
            word_frequency.setdefault(word, 0)
            word_frequency[word] += 1
            word_degree.setdefault(word, 0)
            word_degree[word] += word_list_degree
    for item in word_frequency:
        word_degree[item] = word_degree[item] + word_frequency[item]

    word_score = {}
    for item in word_frequency:
        word_score[item] = word_degree[item] / (word_frequency[item] * 1.0)
    return word_score

def generate_candidate_keyword_scores(phrase_list, word_score):
    keyword_candidates = {}
    for phrase in phrase_list:
        keyword_candidates.setdefault(phrase, 0)
        word_list = separate_words(phrase, 0)
        candidate_score = sum(word_score[word] for word in word_list)
        keyword_candidates[phrase] = candidate_score
    return keyword_candidates

def get_keywords(text, stop_words_path):
    stop_words_pattern = build_stop_word_regex(stop_words_path)
    sentence_list = split_sentences(text)
    phrase_list = generate_candidate_keywords(sentence_list, stop_words_pattern)
    word_scores = calculate_word_scores(phrase_list)
    keyword_candidates = generate_candidate_keyword_scores(phrase_list, word_scores)
    sorted_keywords = sorted(keyword_candidates.items(), key=operator.itemgetter(1), reverse=True)
    top_keywords = sorted_keywords[:15]  # Get the top 15 keywords
    return top_keywords



content = """. Introduction
Erode Venkatappa Ramasamy[1] (17 September 1879 – 24 December 1973), commonly known as Periyar or Thanthai Periyar, was an Indian social activist and politician who started the Self-Respect Movement and Dravidar Kazhagam. He is known as the 'Father of the Dravidian movement'.[2] He rebelled against Brahminical dominance and gender and caste inequality in Tamil Nadu.[3][4][5]

E.V. Ramasamy joined the Indian National Congress in 1919, but resigned in 1925 when he felt that the party was only serving the interests of Brahmins. He questioned the subjugation of non-Brahmin Dravidians as Brahmins enjoyed gifts and donations from non-Brahmins but opposed and discriminated against non-Brahmins in cultural and religious matters.[6][7] In 1924, E.V. Ramasamy participated in non-violent agitation (satyagraha) in Vaikom, Travancore.[8] From 1929 to 1932 Ramasamy made a tour of British Malaya, Europe, and Soviet Union which influenced him.[9][10][11] In 1939, E.V. Ramasamy became the head of the Justice Party,[12] and in 1944, he changed its name to Dravidar Kazhagam.[13] The party later split with one group led by C. N. Annadurai forming the Dravida Munnetra Kazhagam (DMK) in 1949.[13] While continuing the Self-Respect Movement, he advocated for an independent Dravida Nadu (land of the Dravidians).[14]

E.V. Ramasamy promoted the principles of rationalism, self-respect, women’s rights and eradication of caste. He opposed the exploitation and marginalisation of the non-Brahmin Dravidian people of South India and the imposition of what he considered Indo-Aryan India.

2. Biography
2.1. Early Years

B. R. Ambedkar with Periyar when they met in connection with a Buddhist conference in Rangoon, Myanmar in 1954. https://handwiki.org/wiki/index.php?curid=1324644
Erode Venkata Ramasamy was born on 17 September 1879 to a Kannada[15] Balija merchant family[16][17][18] in Erode, then a part of the Coimbatore district of the Madras Presidency.[19] E. V. Ramasamy's father was Venkatappa Nayakar (or Venkata), and his mother was Chinnathyee, Muthammal. He had one elder brother named Krishnaswamy and two sisters named Kannamma and Ponnuthoy.[1][19] He later came to be known as "Periyar" meaning 'respected one' or 'elder' in the Tamil.[1][20][21][22][23]

E. V. Ramasamy married when he was 19, and had a daughter who lived for only 5 months. His first wife, Nagammai, died in 1933.[24] E.V. Ramasamy married for a second time in July 1948.[25] His second wife, Maniammai, continued E. V. Ramasamy's social work after his death in 1973, and his ideas then were advocated by Dravidar Kazhagam.[26]

In 1929, E. V. Ramasamy announced the deletion of his caste title Naicker from his name at the First Provincial Self-Respect Conference of Chengalpattu.[27] He could speak two Dravidian languages: Kannada and Tamil.[28][29][30][31]

[32] Periyar attended school for five years after which he joined his father's trade at the age of 12. He used to listen to Tamil Vaishnavite gurus who gave discourses in his house enjoying his father's hospitality. At a young age, he began questioning the apparent contradictions in the Hindu mythological stories.[1] As Periyar grew, he felt that people used religion only as a mask to deceive innocent people and therefore took it as one of his duties in life to warn people against superstitions and priests.[33] 
E.V. Ramasamy's father arranged for his wedding when he was nineteen. The bride, Nagammai, was only thirteen. Despite having an arranged marriage, Periyar and Nagammai were already in love with each other. Nagammai actively supported her husband in his later public activities and agitation. Two years after their marriage, a daughter was born to them. However, their daughter died when she was five months old. The couple had no more children.[24]

2.2. Kashi Pilgrimage Incident
In 1904, E.V. Ramasamy went on a pilgrimage to Kashi to visit the revered Shiva temple of Kashi Vishwanath.[1] Though regarded as one of the holiest sites of Hinduism, he witnessed immoral activities such as begging, and floating dead bodies.[1] His frustrations extended to functional Hinduism in general when he experienced what he called Brahmanic exploitation.[34]

However, one particular incident in Kasi had a profound impact on E.V. Ramasamy's ideology and future work. At the worship site there were free meals offered to guests. To E.V. Ramasamy's shock, he was refused meals at choultries, which exclusively fed Brahmins. Due to extreme hunger, E.V. Ramasamy felt compelled to enter one of the eateries disguised as a Brahmin with a sacred thread on his bare chest, but was betrayed by his moustache. The gatekeeper at the temple concluded that E.V. Ramasamy was not a Brahmin, as Brahmins were not permitted by the Hindu shastras to have moustaches. He not only prevented Periyar's entry but also pushed him rudely into the street.[1]

As his hunger became intolerable, Periyar was forced to feed on leftovers from the streets. Around this time, he realised that the eatery which had refused him entry was built by a wealthy non-Brahmin from South India.[1] This discriminatory attitude dealt a blow to Periyar's regard for Hinduism, for the events he had witnessed at Kasi were completely different from the picture of Kasi he had in mind, as a holy place which welcomed all.[1] Ramasamy was a theist until his visit to Kasi, after which his views changed and he became an atheist.[35]

2.3. Member of Congress Party (1919–1925)

E.V. Ramasamy statue at Vaikom town in Kottayam, Kerala https://handwiki.org/wiki/index.php?curid=1090103
E.V. Ramasamy joined the Indian National Congress in 1919 after quitting his business and resigning from public posts. He held the chairmanship of Erode Municipality and wholeheartedly undertook constructive programs spreading the use of Khadi, picketing toddy shops, boycotting shops selling foreign cloth, and eradicating untouchability. In 1921, Periyar courted imprisonment for picketing toddy shops in Erode. When his wife as well as his sister joined the agitation, it gained momentum, and the administration was forced to come to a compromise. He was again arrested during the Non-Cooperation movement and the Temperance movement.[6] In 1922, Periyar was elected the President of the Madras Presidency Congress Committee during the Tirupur session, where he advocated strongly for reservation in government jobs and education. His attempts were defeated in the Congress party due to discrimination and indifference, which led to his leaving the party in 1925.[7]

2.4. Vaikom Satyagraha (1924–1925)
According to the prevalent caste system in Kerala and the rest of India, low-caste Hindus were denied entry into temples. In Kerala, they were denied permission to walk on the roads that led to the temples also. (Kerala state was formed in 1956; earlier it was broadly divided into Malabar (North Kerala), Cochin and Travancore kingdoms).

In the Kakinada meet of the Congress Party in 1923, T K Madhavan presented a report citing the discrimination faced by the depressed castes in Kerala. That session decided to promote movements against untouchability.

In Kerala, a committee was formed comprising people of different castes to fight untouchability in the region. The committee was chaired by K Kelappan; the rest of the members were T K Madhavan, Velayudha Menon, Kurur Neelakantan Namboodiripad and T R Krishnaswami Iyer. In early 1924, they decided to launch a ‘Keralaparyatanam’ to gain temple entry and also the right to use public roads for every Hindu irrespective of caste or creed.

The movement gained all-India prominence and support came from far and wide. The Akalis of Punjab lend their support by setting up kitchens to provide food to the Satyagrahis. Even Christian and Muslim leaders came forward for support. This was shunned by Gandhiji who wanted the movement to be an intra-Hindu affair. On advice from Gandhiji, the movement was withdrawn temporarily in April 1924. After the talks with caste-Hindus failed, the leaders resumed the movement. Leaders T K Madhavan and K P Kesava Menon were arrested. Periyar came from Tamil Nadu to give his support. He was arrested.

On 1 October 1924, a group of savarnas (forward castes) marched in a procession and submitted a petition to the Regent Maharani Sethulakshmi Bai of Travancore with about 25000 signatures for temple entry to everyone. Gandhiji also met with the Regent Maharani. This procession of savarnas was led by Mannath Padmanabhan Nair. Starting with about 500 people at Vaikom, the number increased to about 5000 when the procession reached Thiruvananthapuram in November 1924.

In February 1924, they decided to launch a ‘Keralaparyatanam’ to gain temple entry and also the right to use public roads for every Hindu irrespective of caste or creed.

In Vaikom, a small town in Kerala state, then Travancore, there were strict laws of untouchability in and around the temple area. Dalits, also known as Harijans, were not allowed into the close streets around and leading to the temple, let alone inside it. Anti-caste feelings were growing and in 1924 Vaikom was chosen as a suitable place for an organised Satyagraha. Under his guidance a movement had already begun with the aim of giving all castes the right to enter the temples. Thus, agitations and demonstrations took place. On 14 April, Periyar and his wife Nagamma arrived in Vaikom. They were immediately arrested and imprisoned for participation. In spite of Gandhi's objection to non-Keralites and non-Hindus taking part, Periyar and his followers continued to give support to the movement until it was withdrawn. He received the title Vaikom Veeran, given by his followers who participated in the Satyagraha.[36][37][38]

The way in which the Vaikom Satyagraha events have been recorded provides a clue to the image of the respective organisers. In an article entitle Gandhi and Ambedkar, A Study in Leadership, Eleanor Zelliot relates the 'Vaikom Satyagraha', including Gandhi's negotiations with the temple authorities in relation to the event. Furthermore, the editor of E.V. Ramasamy's Thoughts states that Brahmins purposely suppressed news about E.V. Ramasamy's participation. A leading Congress magazine, Young India, in its extensive reports on Vaikom never mentions E.V. Ramasamy.[34]

In Kerala, a committee was formed comprising people of different castes to fight untouchability in the region. The committee chaired by K Kelappan, composed of T K Madhavan, Velayudha Menon, Kurur Neelakantan Namboodiripad and T R Krishnaswami Iyer. In February 1924, they decided to launch a ‘Keralaparyatanam’ to gain temple entry and also the right to use public roads for every Hindu irrespective of caste or creed.

2.5. Self-Respect Movement

Periyar during the early years of Self-Respect Movement https://handwiki.org/wiki/index.php?curid=1668674
Periyar and his followers campaigned constantly to influence and pressure the government to take measures to remove social inequality,(abolish untouchability, manual scavenging system etc) even while other nationalist forerunners focused on the struggle for political independence. The Self-Respect Movement was described from the beginning as "dedicated to the goal of giving non-Brahmins a sense of pride based on their Dravidian past".[39]

In 1952, the Periyar Self-Respect Movement Institution was registered with a list of objectives of the institution from which may be quoted as

for the diffusion of useful knowledge of political education; to allow people to live a life of freedom from slavery to anything against reason and self respect; to do away with needless customs, meaningless ceremonies, and blind superstitious beliefs in society; to put an end to the present social system in which caste, religion, community and traditional occupations based on the accident of birth, have chained the mass of the people and created "superior" and "inferior" classes... and to give people equal rights; to completely eradicate untouchability and to establish a united society based on brother/sisterhood; to give equal rights to women; to prevent child marriages and marriages based on law favourable to one sect, to conduct and encourage love marriages, widow marriages, inter caste and inter-religious marriages and to have the marriages registered under the Civil Law; and to establish and maintain homes for orphans and widows and to run educational institutions.[34]

Propagation of the philosophy of self respect became the full-time activity of Periyar since 1925. A Tamil weekly Kudi Arasu started in 1925, while the English journal Revolt started in 1928 carried on the propaganda among the English educated people.[40] The Self-Respect Movement began to grow fast and received the sympathy of the heads of the Justice Party from the beginning. In May 1929, a conference of Self-Respect Volunteers was held at Pattukkotai under the presidency of S. Guruswami. K.V. Alagiriswami took charge as the head of the volunteer band. Conferences followed in succession throughout the Tamil districts of the former Madras Presidency. A training school in Self-Respect was opened at Erode, the home town of Periyar. The object was not just to introduce social reform but to bring about a social revolution to foster a new spirit and build a new society.[41]

2.6. International Travel (1929–1932)
Between 1929 and 1935, under the strain of World Depression, political thinking worldwide received a jolt from the spread of international communism.[10] Indian political parties, movements and considerable sections of leadership were also affected by inter-continental ideologies. The Self-Respect Movement also came under the influence of the leftist philosophies and institutions. E.V. Ramasamy, after establishing the Self-Respect Movement as an independent institution, began to look for ways to strengthen it politically and socially. To accomplish this, he studied the history and politics of different countries, and personally observed these systems at work.[10]

E.V. Ramasamy toured Malaya for a month, from December 1929 to January 1930, to propagate the self-respect philosophy. Embarking on his journey from Nagapattinam with his wife Nagammal and his followers, E.V. Ramasamy was received by 50,000 Tamil Malaysians in Penang. During the same month, he inaugurated the Tamils Conference, convened by the Tamils Reformatory Sangam in Ipoh, and then went to Singapore. In December 1931 he undertook a tour of Europe, accompanied by S. Ramanathan and Erode Ramu, to personally acquaint himself with their political systems, social movements, way of life, economic and social progress and administration of public bodies. He visited Egypt, Greece, Turkey, the Soviet Union, Germany, England, Spain, France and Portugal, staying in Russia for three months. On his return journey he halted at Ceylon and returned to India in November 1932.[10]

The tour shaped the political ideology of E.V. Ramasamy to achieve the social concept of Self-Respect. The communist system in the Soviet Union appealed to him as appropriately suited to deal with the social ills of the country. Thus, on socio-economic issues Periyar was Marxist, but he did not advocate for abolishing private ownership.[42] Immediately after his return, E.V. Ramasamy, in alliance with the enthusiastic communist, M. Singaravelar, began to work out a socio-political scheme incorporating socialist and self-respect ideals. This marked a crucial stage of development in the Self-Respect Movement which got politicised and found its compatibility in Tamil Nadu.[10]

2.7. Opposition to Hindi
In 1937, when Chakravarthi Rajagopalachari became the Chief Minister of Madras Presidency, he introduced Hindi as a compulsory language of study in schools, thereby igniting a series of anti-Hindi agitations.[14] Tamil nationalists, the Justice Party under Sir A. T. Panneerselvam, and E.V. Ramasamy organised anti-Hindi protests in 1938 which ended with numerous arrests by the Rajaji government.[43]

During the same year, the slogan "Tamil Nadu for Tamilians"[44] was first used by E.V. Ramasamy in protest against the introduction of Hindi in schools. He claimed that the introduction of Hindi was a dangerous mechanism used by the Aryans to infiltrate Dravidian culture.[44] He reasoned that the adoption of Hindi would make Tamils subordinate to the Hindi-speaking North Indians. E.V. Ramasamy claimed that Hindi would not only halt the progress of Tamil people, but would also completely destroy their culture and nullify the progressive ideas that had been successfully inculcated through Tamil in the recent decades.[45]

Cutting across party lines, South Indian politicians rallied together in their opposition to Hindi.[45] There were recurrent anti-Hindi agitations in 1948, 1952 and 1965.[46]

2.8. As President of the Justice Party (1938–1944)
A political party known as the South Indian Libertarian Federation (commonly referred to as Justice Party) was founded in 1916, principally to oppose the economic and political power of the Brahmin groups. The party's goal was to render social justice to the non-Brahmin groups. To gain the support of the masses, non-Brahmin politicians began propagating an ideology of equality among non-Brahmin castes. Brahmanical priesthood and Sanskritic social class-value hierarchy were blamed for the existence of inequalities among non-Brahmin caste groups.[13]

In 1937, when the government required that Hindi be taught in the school system, E.V. Ramasamy organised opposition to this policy through the Justice Party. After 1937, the Dravidian movement derived considerable support from the student community. In later years, opposition to Hindi played a big role in the politics of Tamil Nadu. The fear of the Hindi language had its origin in the conflict between Brahmins and non-Brahmins. To the Tamils, acceptance of Hindi in the school system was a form of bondage. When the Justice Party weakened in the absence of mass support, E.V. Ramasamy took over the leadership of the party after being jailed for opposing Hindi in 1939.[12] Under his tutelage the party prospered, but the party's conservative members, most of whom were rich and educated, withdrew from active participation.[13]

2.9. Dravidar Kazhagam (1944–onwards)
Formation of the Dravidar Kazhagam
At a rally in 1944, Periyar, in his capacity as the leader of the Justice Party, declared that the party would henceforth be known as the Dravidar Kazhagam, or "Dravidian Association". However, a few who disagreed with Periyar started a splinter group, claiming to be the original Justice Party. This party was led by veteran Justice Party leader P. T. Rajan and survived until 1957.

The Dravidar Kazhagam came to be well known among the urban communities and students. Villages were influenced by its message. Hindi, and ceremonies that had become associated with Brahmanical priesthood, were identified as alien symbols that should be eliminated from Tamil culture. Brahmins, who were regarded as the guardians of such symbols, came under verbal attack.[13] From 1949 onwards, the Dravidar Kazhagam intensified social reformist work and put forward the fact that superstitions were the cause for the degeneration of Dravidians. The Dravidar Kazhagam vehemently fought for the abolition of untouchability amongst the Dalits. It also focused its attention on the liberation of women, women's education, willing marriage, widow marriage, orphanages and mercy homes.[47]

Split with Annadurai
In 1949, E.V. Ramasamy's chief lieutenant, Conjeevaram Natarajan Annadurai, established a separate association called the Dravida Munnetra Kazhagam (DMK), or Dravidian Progressive Federation.[13] This was due to differences between the two, while Periyar advocated a separate independent Dravidian or Tamil state, Annadurai compromised with the Delhi government, at the same time claiming increased state independence.[48] E.V. Ramasamy was convinced that individuals and movements that undertake the task of eradicating the social evils in the Indian sub-continent have to pursue the goal with devotion and dedication without deviating from the path and with uncompromising zeal. Thus, if they contest elections aiming to assume political power, they would lose vigour and a sense of purpose. But among his followers, there were those who had a different view, wanting to enter into politics and have a share in running the government. They were looking for an opportunity to part with E.V. Ramasamy. Thus, when E.V. Ramasamy married Maniammai on 9 July 1948, they quit the Dravidar Kazhagam, stating that E.V. Ramasamy married Maniammayar who was the daughter of Kanagasabhai when he was 70 and she 32. Those who parted company with E.V. Ramasamy joined the DMK.[25] Though the DMK split from the Dravidar Kazhagam, the organisation made efforts to carry on E.V. Ramasamy's Self-Respect Movement to villagers and urban students. The DMK advocated the thesis that the Tamil language was much richer than Sanskrit and Hindi in content, and thus was a key which opened the door to subjects to be learned.[13] The Dravidar Kazhagam continued to counter Brahminism, Indo-Aryan propaganda, and uphold the Dravidians' right of self-determination.[49]

Later years

Periyar Thidal at Vepery, where Periyar's body was buried. https://handwiki.org/wiki/index.php?curid=1776585
In 1956, despite warnings from P. Kakkan, the President of the Tamil Nadu Congress Committee, Periyar organised a procession to the Marina to burn pictures of the Hindu God Rama.[50] Periyar was subsequently arrested and confined to prison.[50]

The activities of Periyar continued when he went to Bangalore in 1958 to participate in the All India Official Language Conference. There he stressed the need to retain English as the Union Official Language instead of Hindi. Five years later, Periyar travelled to North India to advocate the eradication of the caste system. In his last meeting at Thiagaraya Nagar, Chennai on 19 December 1973, Periyar declared a call for action to gain social equality and a dignified way of life. On 24 December 1973, Periyar died at the age of 94.[25]

3. Principles and Legacy
Periyar spent over fifty years giving speeches, propagating the realisation that everyone is an equal citizen and the differences on the basis of caste and creed were man-made to keep the innocent and ignorant as underdogs in the society. Although Periyar's speeches were targeted towards the illiterate and more mundane masses, scores of educated people were also swayed.[51] Periyar viewed reasoning as a special tool. According to him, all were blessed with this tool, but very few used it. Thus Periyar used reasoning with respect to subjects of social interest in his presentations to his audiences.[51] Communal differences in Tamil society were considered by many to be deep-rooted features until Periyar came to the scene.[52]

3.1. Rationalism
The bedrock of E.V. Ramasamy’s principles and the movements that he started was rationalism. He thought that an insignificant minority in society was exploiting the majority and trying to keep it in a subordinate position forever. He wanted the exploited to sit up and think about their position, and use their reason to realise that they were being exploited by a handful of people. If they started thinking, they would realise that they were human beings like the rest, that birth did not and should not endow superiority over others and that they must awaken themselves and do everything possible to improve their own lot.[51]

Likewise, E.V. Ramasamy explained that wisdom lies in thinking and that the spear-head of thinking is rationalism. On caste, he stated that no other living being harms or degrades its own class. But man, said to be a rational living being, does these evils. The differences, hatred, enmity, degradation, poverty, and wickedness, now prevalent in the society are due to lack of wisdom and rationalism and not due to God or the cruelty of time. E.V. Ramasamy had written in his books and magazines dozens of times of various occasions that the British rule is better than self-rule[53]

E.V. Ramasamy also blamed the capitalists for their control of machineries, creating difficulties for the workers. According to his philosophy, rationalism, which has to lead the way for peaceful life to all, had resulted in causing poverty and worries to the people because of dominating forces. He stated that there is no use of simply acquiring titles or amassing wealth if one has no self-respect or scientific knowledge. An example he gave was the West sending messages to the planets, while the Tamil society in India were sending rice and cereals to their dead forefathers through the Brahmins.[53]

In a message to the Brahmin community, Periyar stated, "in the name of god, religion, and sastras you have duped us. We were the ruling people. Stop this life of cheating us from this year. Give room for rationalism and humanism".[54] He added that "any opposition not based on rationalism, science, or experience will one day or another, reveal the fraud, selfishness, lies and conspiracies".[54]

3.2. Self-respect
Periyar's philosophy of self-respect was based on his image of an ideal world and a universally accepted one. His philosophy preaches that human actions should be based on rational thinking. Further, the outcome of the natural instinct of human beings is to examine every object and every action and even nature with a spirit of inquiry, and to refuse to submit to anything irrational as equivalent to slavery. Thus, the philosophy of self-respect taught that human actions should be guided by reason, right and wrong should follow from rational thinking and conclusions drawn from reason should be respected under all circumstances. Freedom means respect to thoughts and actions considered 'right' by human beings on the basis of 'reason'. There is not much difference between 'freedom' and 'self-respect'.[55]

Periyar's foremost appeal to people was to develop self-respect. He preached that the Brahmins had monopolised and cheated other communities for decades and deprived them of self-respect. He stated that most Brahmins claimed to belong to a "superior" community with the reserved privilege of being in charge of temples and performing archanas. He felt that they were trying to reassert their control over religion by using their superior caste status to claim the exclusive privilege to touch idols or enter the sanctum sanctorum.[52]

3.3. Women’s Rights
As a rationalist and ardent social reformer, Periyar advocated forcefully throughout his life that women should be given their legitimate position in society as the equals of men and that they should be given good education and have the right to property. He thought age and social customs was not a bar in marrying women. He was keen that women should realise their rights and be worthy citizens of their country.[56]

Periyar fought against the orthodox traditions of marriage as suppression of women in Tamil Nadu and throughout the Indian sub-continent. Though arranged marriages were meant to enable a couple to live together throughout life, it was manipulated to enslave women.[57] Much worse was the practice of child marriages practised throughout India at the time. It was believed that it would be a sin to marry after puberty.[58] Another practice, which is prevalent today, is the dowry system where the bride's family is supposed to give the husband a huge payment for the bride. The purpose of this was to assist the newly wedded couple financially, but in many instances dowries were misused by bridegrooms. The outcome of this abuse turned to the exploitation of the bride's parents wealth, and in certain circumstances, lead to dowry deaths.[59] There have been hundreds of thousands of cases where wives have been murdered, mutilated, and burned alive because the father of the bride was unable to make the dowry payment to the husband. Periyar fiercely stood up against this abuse meted out against women.[60]

Women in India also did not have rights to their families' or husbands' property. Periyar fought fiercely for this and also advocated for women to have the right to separate or divorce their husbands under reasonable circumstances.[60] While birth control remained taboo in society of Periyar's time, he advocated for it not only for the health of women and population control, but for the liberation of women.[61]

He criticised the hypocrisy of chastity for women and argued that it should either apply also to men, or not at all for both genders.[62] While fighting against this, Periyar advocated getting rid of the Devadasi system. In his view it was an example of a list of degradations of women, attaching them to temples for the entertainment of others, and as temple prostitutes.[63] Further, for the liberation of women, Periyar pushed for their right to have an education and to join the armed services and the police force.[62][64]

According to biographer M.D. Gopalakrishnan, Periyar and his movement achieved a better status for women in Tamil society. Periyar held that, in matters of education and employment, there should be no difference between men and women. Gopalakrishnan states that Periyar's influence in the State departments and even the Center made it possible for women to join police departments and the army. Periyar also spoke out against child marriage.[52]

3.4. Social Reform and Eradication of Caste

Periyar on postal stamp of India, issued in 2009 https://handwiki.org/wiki/index.php?curid=1289791
Periyar wanted thinking people to see their society as far from perfect and in urgent need of reform. He wanted the government, the political parties and social workers to identify the evils in society and boldly adopt measures to remove them.[65] Periyar's philosophy did not differentiate social and political service.[66] According to him, the first duty of a government is to run the social organisation efficiently, and the philosophy of religion was to organise the social system. Periyar stated that while Christian and Islamic religions were fulfilling this role, the Hindu religion remained totally unsuitable for social progress. He argued that the government was not for the people, but, in a "topsy-turvy" manner, the people were for the government. He attributed this situation to the state of the social system contrived for the advantage of a small group of people.[66]

One of the areas of Periyar's focus was on the upliftment of rural communities. In a booklet called Village Uplift, Periyar pleaded for rural reform. At that time rural India still formed the largest part of the Indian subcontinent, in spite of the ongoing process of urbanisation. Thus, the distinction between rural and urban had meant an economic and social degradation for rural inhabitants. Periyar wanted to eradicate the concept of "village" as a discrimination word among places, just as the concept of "outcast" among social groups. Periyar advocated for a location where neither the name nor the situation or its conditions imply differences among people.[67] He further advocated for the modernisation of villages by providing public facilities such as schools, libraries, radio stations, roads, bus transport, and police stations.[68]

Periyar felt that a small number of cunning people created caste distinctions to dominate Indian society, so he emphasised that individuals must first develop self-respect and learn to analyse propositions rationally. According to Periyar, a self-respecting rationalist would readily realise that the caste system had been stifling self-respect and therefore he or she would strive to get rid of this menace.[69]

Periyar stated that the caste system in South India is, due to Indo-Aryan influence, linked with the arrival of Brahmins from the north. Ancient Tamil Nadu (part of Tamilakkam) had a different stratification of society in four or five regions (Tinai), determined by natural surroundings and adequate means of living.[70] Periyar also argued that birds, animals, and worms, which are considered to be devoid of rationalism do not create castes, or differences of high and low in their own species. But man, considered to be a rational being, was suffering from these because of religion and discrimination.[71]

The Samathuvapuram (Equality Village) social equality system introduced by the Government of Tamil Nadu in the late 1990s is named after Ramasamy.[72]

3.5. Tamil Language and Writing
Periyar claimed that Tamil, Telugu, Malayalam, and Kannada came from the same mother language of Old Tamil. He explained that the Tamil language is called by four different names since it is spoken in four different Dravidian states. Nevertheless, current understanding of Dravidian languages contradicts such claims. For example, the currently known classification of Dravidian languages provides the following distinct classes: Southern (including Tamil–Malayalam, Kannada and Tulu); Central (including Telugu–Kui and Kolami–Parji); and, Northern (including Kurukh–Malto and Brahui).

With relation to writing, Periyar stated that using the Tamil script about the arts, which are useful to the people in their life and foster knowledge, talent and courage, and propagating them among the masses, will enlighten the people. Further, he explained that it will enrich the language, and thus it can be regarded as a zeal for Tamil.[73] Periyar also stated that if words of North Indian origin (Sanskrit) are removed from Telugu, Kannada, and Malayalam, only Tamil will be left. On the Brahmin usage of Tamil, he stated that the Tamil spoken by the Andhras and the Malayali people was far better than the Tamil spoken by the Brahmins. Periyar believed that Tamil language will make the Dravidian people unite under the banner of Tamil culture, and that it will make the Kannadigas, Andhras and the Malayalees be vigilant. With regards to a Dravidian alliance under a common umbrella language, Periyar stated that "a time will come for unity. This will go on until there is an end to the North Indian domination. We shall reclaim an independent sovereign state for us".[74]

At the same time, Periyar was also known to have made controversial remarks on the Tamil language and people from time to time. On one occasion, he referred to the Tamil people as "barbarians"[75] and the Tamil language as the "language of barbarians".[75][76][77][78][79] However, Anita Diehl explains that Periyar made these remarks on Tamil because it had no respective feminine verbal forms.[34] But Anita Diehl's explanation doesn't match with Periyar's own explanation. Periyar himself explained reasons many times in his speeches and writings, for instance, an excerpt from his book Thamizhum, Thamizharum(Tamil and Tamil people) reads, "I say Tamil as barbarian language. Many get angry with me for saying so. But no one ponders over why I say so. They say Tamil is a 3,000 to 4,000 years-old language and they boast about this. Precisely that is what the reason why I call Tamil as barbarian language. People should understand the term primitive and barbarism. What was the status of people living 4,000 years ago and now? We are just blindly sticking to old glories. No one has come forward to reform Tamil language and work for its growth."[80]

Periyar's ideas on Tamil alphabet reforms included those such as the reasons for the vowel 'ஈ' (i) having a cursive and looped representation of the short form 'இ' (I).[clarification needed] In stone inscriptions from 400 or 500 years ago, many Tamil letters are found in other shapes. As a matter of necessity and advantage to cope with printing technology, Periyar thought that it was sensible to change a few letters, reduce the number of letters, and alter a few signs. He further explained that the older and more divine a language and its letters were said to be, the more they needed reform. Because of changes brought about by means of modern transport and international contact, and happenings that have attracted words and products from many countries, foreign words and their pronunciations have been assimilated into Tamil quite easily. Just as a few compound characters have separate signs to indicate their length as in ' கா ', ' கே ' (kA:, kE:), Periyar questioned why other compound characters like ' கி ', ' கீ ', 'கு ', ' கூ ' (kI, ki:, kU, ku:) (indicated integrally as of now), shouldn't also have separate signs. Further, changing the shape of letters, creating new symbols and adding new letters and similarly, dropping those that are redundant, were quite essential according to Periyar. Thus, the glory and excellence of a language and its script depend on how easily they can be understood or learned and on nothing else"[34]

3.6. Thoughts on the Thirukkural
Periyar hailed the Thirukkural as a valuable scripture which contained many scientific and philosophical truths. He also praised the secular nature of the work. Periyar praised Thiruvalluvar for his description of God as a formless entity with only positive attributes. He also suggested that one who reads the Thirukkural will become a Self-respecter, absorbing knowledge in politics, society, and economics. According to him, though certain items in this ancient book of ethics may not relate to today, it permitted such changes for modern society.[81]

On caste, he believed that the Kural illustrates how Vedic laws of Manu were against the Sudras and other communities of the Dravidian race. On the other hand, Periyar opined that the ethics from the Kural was comparable to the Christian Bible. The Dravidar Kazhagam adopted the Thirukkural and advocated that Thiruvalluvar's Kural alone was enough to educate the people of the country.[81] One of Periyar's quotes on the Thirukkural from Veeramani's Collected Works of Periyar was "when Dravida Nadu (Dravidistan) was a victim to Indo-Aryan deceit, Thirukkural was written by a great Dravidian Thiruvalluvar to free the Dravidians".[81]

Periyar also asserted that due to the secular nature of Thirukkural, it has the capacity to be the common book of faith for all humanity and can be kept on par or above the holy books of all religions.

3.7. Self-determination of Dravida Nadu

Periyar with Muhammad Ali Jinnah and B. R. Ambedkar https://handwiki.org/wiki/index.php?curid=1098517
The Dravidian-Aryan conflict was believed to be a continuous historical phenomenon that started when the Aryans first set their foot in the Dravidian lands. Even a decade before the idea of separation appeared, Periyar stated that, "as long as Aryan religion, Indo-Aryan domination, propagation of Aryan Vedas and Aryan "Varnashrama" existed, there was need for a "Dravidian Progressive Movement" and a "Self-Respect Movement".[82] Periyar became very concerned about the growing North Indian domination over the south which appeared to him no different from foreign domination. He wanted to secure the fruits of labour of the Dravidians to the Dravidians, and lamented that fields such as political, economic, industrial, social, art, and spiritual were dominated by the north for the benefit of the North Indians. Thus, with the approach of independence from Britain, this fear that North India would take the place of Britain to dominate South India became more and more intense.[83]

Periyar was clear about the concept of a separate nation, comprising Tamil areas, that is part of the then existing Madras Presidency with adjoining areas into a federation guaranteeing protection of minorities, including religious, linguistic, and cultural freedom of the people. A separatist conference was held in June 1940 at Kanchipuram when Periyar released the map of the proposed Dravida Nadu, but failed to get British approval. On the contrary, Periyar received sympathy and support from people such as Bhimrao Ramji Ambedkar and Muhammad Ali Jinnah for his views on the Congress, and for his opposition to Hindi. They then decided to convene a movement to resist the Congress.[82][84]

The concept of Dravida Nadu was later modified down to Tamil Nadu.[85] This led to a proposal for a union of the Tamil people of not only South India but including those of Ceylon as well.[86] In 1953, Periyar helped to preserve Madras as the capital of Tamil Nadu, which later was the name he substituted for the more general Dravida Nadu.[87] In 1955 Periyar threatened to burn the national flag, but on Chief Minister Kamaraj's pledge that Hindi should not be made compulsory, he postponed the action.[34] In his speech of 1957 called Suthantara Tamil Nadu En? (Why an independent Tamil Nadu?), he criticised the Central Government of India, inducing thousands of Tamilians to burn the constitution of India. The reason for this action was that Periyar held the Government responsible for maintaining the caste system. After stating reasons for separation and turning down opinions against it, he closed his speech with a "war cry" to join and burn the map of India on 5 June. Periyar was sentenced to six months imprisonment for burning the Indian constitution.[88]

Advocacy of such a nation became illegal when separatist demands were banned by law in 1957. Regardless of these measures, a Dravida Nadu Separation Day was observed on 17 September 1960 resulting in numerous arrests.[89] However, Periyar resumed his campaign in 1968. He wrote an editorial on 'Tamil Nadu for Tamilians' in which he stated, that by nationalism only Brahmins had prospered and nationalism had been developed to abolish the rights of Tamils. He advocated that there was need to establish a Tamil Nadu Freedom Organization and that it was necessary to work towards it.[90]

3.8. Anti-Brahmanism vs. Anti-Brahmin
Periyar was a radical advocate of anti-Brahmanism. Periyar's ideology of anti-Brahmanism is quite often confused as being anti-Brahmin. Even a non-Brahmin who supports inequality based on caste was seen as a supporter of brahmanism. Periyar called on both Brahmins and non-Brahmins to shun brahmanism.

In 1920, when the Justice Party came to power, Brahmins occupied about 70 percent[26][91] of the high level posts in the government. After reservation was introduced by the Justice Party, it reversed this trend, allowing non-Brahmins to rise in the government of the Madras Presidency.[91] Periyar, through the Justice Party, advocated against the imbalance of the domination of Brahmins who constituted only 3 percent[26][92] of the population, over government jobs, judiciary and the Madras University.[92] His Self-Respect Movement espoused rationalism and atheism and the movement had currents of anti-Brahminism.[93] Furthermore, Periyar stated that:

"Our Dravidian movement does not exist against the Brahmins or the Banias (a North Indian merchant caste). If anyone thinks so, I would only pity him. But we will not tolerate the ways in which Brahminism and the Bandiaism [clarification needed] is degrading Dravidanadu. Whatever support they may have from the government, neither myself nor my movement will be of cowardice".[94][95]

Periyar also criticised Subramanya Bharathi in the journal Ticutar for portraying Mother Tamil as a sister of Sanskrit in his poems:

"They say Bharati is an immortal poet.…Even if a rat dies in an akrakāram, they would declare it to be immortal. … of Tamilnadu praises him. Why should this be so? Supposedly because he sang fulsome praises of Tamil and Tamilnadu. What else could he sing? His own mother tongue, Sanskrit, has been dead for years. What other language did he know? He cannot sing in Sanskrit. … He says Tamilnadu is the land of Aryas."[96]

3.9. Comparisons with Gandhi
In the Vaikom Satyagraha of 1924, Periyar and Gandhi ji both cooperated and confronted each other in socio-political action. Periyar and his followers emphasised the difference in point of view between Gandhi and himself on the social issues, such as fighting the Untouchability Laws and eradication of the caste system.

According to the booklet "Gandhi and Periyar", Periyar wrote in his paper Kudi Arasu in 1925, reporting on the fact that Gandhi was ousted from the Mahasabha because he opposed resolutions for the maintaining of caste and Untouchability Laws which would spoil his efforts to bring about Hindu-Muslim unity. From this, Gandhi learned the need for pleasing the Brahmins if anything was to be achieved.[97]

Peiryar in his references to Gandhi used opportunities to present Gandhi as, on principle, serving the interests of the Brahmins. In 1927, Periyar and Gandhi met at Bangalore to discuss this matter. The main difference between them came out when Periyar stood for the total eradication of Hinduism to which Gandhi objected saying that Hinduism is not fixed in doctrines but can be changed. In the Kudi Arasu, Periyar explained that:

"With all his good qualities, Gandhi did not bring the people forward from foolish and evil ways. His murderer was an educated man. Therefore nobody can say this is a time of high culture. If you eat poison, you will die. If electricity hits the body, you will die. If you oppose the Brahmin, you will die. Gandhi did not advocate the eradication of Varnasrama Dharma structure, but sees in it a task for the humanisation of society and social change possible within its structure. The consequence of this would be continued high-caste leadership. Gandhi adapted Brahmins to social change without depriving them of their leadership".[97]

Gandhi accepted karma in the sense that "the Untouchables reap the reward of their karma,[97] but was against discrimination against them using the revaluing term Harijans. As shown in the negotiations at Vaikom his methods for abolishing discrimination were: to stress on the orthodox, inhumane treatment of Untouchables; to secure voluntary lifting of the ban by changing the hearts of caste Hindus; and to work within a Hindu framework of ideas.[97]

On the Temple Entry issue, Gandhi never advocated the opening of Garbha Griha to Harijans in consequence of his Hindu belief. These sources which can be labelled "pro-Periyar" with the exception of M. Mahar and D.S. Sharma, clearly show that Periyar and his followers emphasised that Periyar was the real fighter for the removal of Untouchability and the true upliftment of Hairjans, whereas Gandhi was not. This did not prevent Periyar from having faith in Gandhi on certain matters.[97]

3.10. Religion and Atheism
Periyar was generally regarded as a pragmatic propagandist who attacked the evils of religious influence on society, mainly what he regarded as Brahmin domination. At a young age, he felt that some people used religion only as a mask to deceive innocent people and regarded it as his life's mission to warn people against superstitions and priests.[33] Anita Diehl explains that Periyar cannot be called an atheist philosopher. Periyar, however, qualified what the term "atheist" implies in his address on philosophy. He repudiated the term as without real sense: "….the talk of the atheist should be considered thoughtless and erroneous. The thing I call god... that makes all people equal and free, the god that does not stop free thinking and research, the god that does not ask for money, flattery and temples can certainly be an object of worship. For saying this much I have been called an atheist, a term that has no meaning".

Anita Diehl explains that Periyar saw faith as compatible with social equality and did not oppose religion itself.[98] In a book on revolution published in 1961, Periyar stated: "be of help to people. Do not use treachery or deceit. Speak the truth and do not cheat. That indeed is service to God."[99]

On Hinduism, Periyar believed that it was a religion with no distinctive sacred book (Bhagawad Gita) or origins, but an imaginary faith preaching the "superiority" of the Brahmins, the inferiority of the Shudras, and the untouchability of the Dalits (Panchamas).[44] Maria Misra, a lecturer at Oxford University, compares him to the philosophes, stating: "his contemptuous attitude to the baleful influence of Hinduism in Indian public life is strikingly akin to the anti-Catholic diatribes of the enlightenment philosophes".[100] In 1955 Periyar was arrested for his public action of burning pictures of Rama in public places as a symbolic protest against the Indo-Aryan domination and degradation of the Dravidian leadership according to the Ramayana epic.[101] Periyar also shoed images of Krishna and Rama, stating that they were Aryan gods that considered the Dravidian Shudras to be "sons of prostitutes".[102]

Periyar openly suggested to those who were marginalised within the Hindu communities to consider converting to other faiths such as Islam, Christianity, or Buddhism. On Islam, he stated how it was good for abolishing the disgrace in human relationship, based on one of his speeches to railway employees at Tiruchirapalli in 1947. Periyar also commended Islam for its belief in one invisible and formless God; for proclaiming equal rights for men and women; and for advocating social unity.[103]

At the rally in Tiruchi, Periyar said:

"Muslims are following the ancient philosophies of the Dravidians. The Arabic word for Dravidian religion is Islam. When Brahmanism was imposed in this country, it was Mohammad Nabi who opposed it, by instilling the Dravidian religion's policies as Islam in the minds of the people"[104]

Periyar viewed Christianity as similar to the monotheistic faith of Islam. He explained that the Christian faith says that there can be only one God which has no name or shape. Periyar took an interest in Martin Luther - both he and his followers wanted to liken him and his role to that of the European reformer. Thus Christian views, as expressed for example in The Precepts of Jesus (1820) by Ram Mohan Roy, had at least an indirect influence on Periyar.[105]

Apart from Islam and Christianity, Periyar also found in Buddhism a basis for his philosophy, though he did not accept that religion. It was again an alternative in the search for self-respect and the object was to get liberation from the discrimination of Hinduism.[106] Through Periyar's movement, Temple Entry Acts of 1924, 1931, and up to 1950 were created for non-Brahmins. Another accomplishment took place during the 1970s when Tamil replaced Sanskrit as the temple language in Tamil Nadu, while Dalits finally became eligible for priesthood.[34]

3.11. Controversies
Factionism in the Justice Party
When B. Munuswamy Naidu became the Chief Minister of Madras Presidency in 1930, he endorsed the inclusion of Brahmins in the Justice Party, saying:

So long as we exclude one community, we cannot as a political party speak on behalf of, or claim to represent all the people of our presidency. If, as we hope, provincial autonomy is given to the provinces as a result of the reforms that may be granted, it should be essential that our Federation should be in a position to claim to be a truly representative body of all communities. What objection can there be to admit such Brahmins as are willing to subscribe to the aims and objects of our Federation? It may be that the Brahmins may not join even if the ban is removed. But surely our Federation will not thereafter be open to objection on the ground that it is an exclusive organisation.[107]

Though certain members supported the resolution, a faction in the Justice Party known as the "Ginger Group" opposed the resolution and eventually voted it down. Periyar, who was then an observer in the Justice Party, criticised Munuswamy Naidu, saying:

At a time when non-Brahmins in other parties were gradually coming over to the Justice Party, being fed up with the Brahmin's methods and ways of dealing with political questions, it was nothing short of folly to think of admitting him into the ranks of the Justice Party.[108]

This factionism continued until 1932 when Munuswamy Naidu stepped down as the Chief Minister of Madras and the Raja of Bobbili became the chief minister.[108]

4. Followers and Influence

Periyar on postal stamp of India, issued in 1978 https://handwiki.org/wiki/index.php?curid=1224043
After the death of Periyar in 1973, conferences were held throughout Tamil Nadu for a week in January 1974. The same year Periyar's wife, Maniyammai, the new head of the Dravidar Kazhagam, set fire to the effigies of 'Rama', 'Sita' and 'Lakshmana' at Periyar Thidal, Madras. This was in retaliation to the Ramaleela celebrations where effigies of 'Ravana', 'Kumbakarna' and 'Indrajit' were burnt in New Delhi. For this act she was imprisoned. During the 1974 May Day meetings held at different places in Tamil Nadu, a resolution urging the Government to preserve 80 percent[26] of jobs for Tamils was passed. Soon after this, a camp was held at Periyar Mansion in Tiruchirapalli to train young men and women to spread the ideals of the Dravidar Kazhagam in rural areas.[26]

On Periyar's birthday on 17 September 1974, Periyar's Rationalist Library and Research Library and Research Institute was opened by the then Tamil Nadu Chief Minister M. Karunanidhi. This library contained Periyar's rationalist works, the manuscripts of Periyar and his recorded speeches.[69] Also during the same year Periyar's ancestral home in Erode, was dedicated as a commemoration building. On 20 February 1977, the opening function of Periyar Building in Madras was held. At the meeting which the Managing Committee of the Dravidar Kazhagam held, there on that day, it was decided to support the candidates belonging to the Janata Party, the Dravida Munnetra Kazhagam (DMK), and the Marxist Party during the General Elections.[26]

On 16 March 1978, Maniyammai died. The Managing Committee of the Dravidar Kazhagam elected K. Veeramani as General Secretary of the Dravidar Kazhagam on 17 March 1978. From then on, the Periyar-Maniyammai Educational and Charitable Society started the Periyar Centenary Women's Polytechnic at Thanjavur on 21 September 1980. On 8 May 1982, the College for Correspondence Education was started under the auspices of the Periyar Rationalist Propaganda Organization.[26]

Over the years, Periyar influenced Tamil Nadu's political party heads such as C.N. Annadurai[25] and M. Karunanidhi[109] of the Dravida Munnetra Kazhagam' (DMK), V. Gopalswamy[110][111] founder of the Marumalarchi Dravida Munnetra Kazhagam (MDMK), S. Ramadoss[112] founder of the Pattali Makkal Katchi (PMK), Thol. Thirumavalavan, founder of the Dalit Panthers of India (DPI), and Dravidar Kazhagam's K. Veeramani.[113] Nationally, Periyar is main ideological icon for India's third largest voted party, Bahujan Samaj Party[114][115] and its founder Kanshi Ram.[116] Other political figures influenced by Periyar were former Congress minister K. Kamaraj,[25] former Chief Minister of Uttar Pradesh Mayawati.[117] Periyar's life and teachings have also influenced writers and poets such as Kavignar Inkulab, and Bharathidasan[118] and actors such as Kamal Haasan[119] and Sathyaraj.[120] Noted Tamil Comedian N. S. Krishnan was a close friend and follower of Periyar.[121][122] W. P. A. Soundarapandian Nadar was a close confidant of Periyar and encouraged Nadars to be a part of the Self-Respect Movement.[123][124] A writer from Uttar Pradesh, Lalai Singh Yadav translated Periyar's notable works into Hindi.[125][126][127]

5. In Popular Culture
Sathyaraj and Khushboo Sundar starred in a government-sponsored film Periyar released in 2007. Directed by Gnana Rajasekaran, the film was screened in Malaysia on 1 May 2007 and was screened at the Goa International Film Festival in November that year.[128][129] Sathyaraj reprised his role as Periyar in the film Kalavadiya Pozhudugal directed by Thangar Bachan which released in 2017.[130]"""
#print(get_keywords(content, "SmartStoplist.txt"))
doc = content
kw_model = KeyBERT()
kw_model = KeyBERT(model='multi-qa-MiniLM-L6-cos-v1')
import subprocess


def create_requirements():
    # Run conda list --export to get package details
    result = subprocess.run(['conda', 'list', '--export'], capture_output=True, text=True)
    packages = result.stdout.splitlines()

    # Open the requirements.txt file for writing
    with open('requirements.txt', 'w') as f:
        for package in packages:
            parts = package.split('=')
            if len(parts) == 3:
                # Write package and version in pip format
                f.write(f"{parts[0]}=={parts[1]}\n")




try:
    create_requirements()
    #keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), stop_words=None)
    #keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(3, 3), stop_words='english',
                              #use_maxsum=True, nr_candidates=20, top_n=5)
    #print(keywords)
except AttributeError as e:
    # Handle the AttributeError gracefully
    print("AttributeError occurred:", e)
    keywords = []  # Provide an empty list of keywords as a fallback
    
    
import subprocess


