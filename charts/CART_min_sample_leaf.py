import matplotlib.pyplot as plt

arr = [(0.7506297229219143, 1, 5.223741292953491, 0.021051883697509766), (0.7758186397984886, 2, 3.540972948074341, 0.01973891258239746), (0.7632241813602015, 3, 2.94213604927063, 0.01951003074645996), (0.7682619647355163, 4, 2.5950918197631836, 0.018604040145874023), (0.7682619647355163, 5, 2.328881025314331, 0.021034955978393555), (0.7707808564231738, 6, 2.0537190437316895, 0.018645048141479492), (0.7732997481108312, 7, 1.8884239196777344, 0.01797628402709961), (0.7783375314861462, 8, 1.7610979080200195, 0.01822185516357422), (0.7783375314861462, 9, 1.6902542114257812, 0.018157005310058594), (0.7783375314861462, 10, 1.57682204246521, 0.017544984817504883), (0.7732997481108312, 11, 1.5124969482421875, 0.01793694496154785), (0.7581863979848866, 12, 1.4606411457061768, 0.01769113540649414), (0.7707808564231738, 13, 1.401392936706543, 0.017361879348754883), (0.783375314861461, 14, 1.3496301174163818, 0.017338037490844727), (0.7758186397984886, 15, 1.3288969993591309, 0.017602205276489258), (0.7682619647355163, 16, 1.2958779335021973, 0.017698049545288086), (0.7632241813602015, 17, 1.263718843460083, 0.017595291137695312), (0.7632241813602015, 18, 1.2568550109863281, 0.01703810691833496), (0.7682619647355163, 19, 1.2296440601348877, 0.018151044845581055), (0.7682619647355163, 20, 1.170295000076294, 0.016999006271362305), (0.7682619647355163, 21, 1.1357800960540771, 0.017171859741210938), (0.7682619647355163, 22, 1.1162478923797607, 0.01783609390258789), (0.7682619647355163, 23, 1.0955870151519775, 0.016929149627685547), (0.7682619647355163, 24, 1.0821237564086914, 0.01676321029663086), (0.7556675062972292, 25, 1.0623958110809326, 0.017159223556518555), (0.7556675062972292, 26, 1.0485341548919678, 0.01667189598083496), (0.7556675062972292, 27, 1.0384790897369385, 0.01720595359802246), (0.7556675062972292, 28, 1.0361428260803223, 0.016862154006958008), (0.7657430730478589, 29, 1.0143060684204102, 0.017032861709594727), (0.7481108312342569, 30, 1.0064582824707031, 0.016682863235473633), (0.7481108312342569, 31, 0.992344856262207, 0.016706228256225586), (0.7481108312342569, 32, 0.9777531623840332, 0.016769886016845703), (0.7481108312342569, 33, 0.9701108932495117, 0.017097949981689453), (0.7455919395465995, 34, 0.9437839984893799, 0.016829967498779297), (0.7632241813602015, 35, 0.9337611198425293, 0.01683497428894043), (0.760705289672544, 36, 0.8850939273834229, 0.01660013198852539), (0.7632241813602015, 37, 0.8624610900878906, 0.016483783721923828), (0.7632241813602015, 38, 0.8562381267547607, 0.016697168350219727), (0.7556675062972292, 39, 0.8459148406982422, 0.01620316505432129), (0.7632241813602015, 40, 0.8295831680297852, 0.016305923461914062), (0.7632241813602015, 41, 0.8251898288726807, 0.015900850296020508), (0.760705289672544, 42, 0.8245360851287842, 0.016022920608520508), (0.760705289672544, 43, 0.8125309944152832, 0.01604604721069336), (0.7657430730478589, 44, 0.8020808696746826, 0.01613903045654297), (0.7657430730478589, 45, 0.800537109375, 0.015894174575805664), (0.7657430730478589, 46, 0.7890548706054688, 0.0163421630859375), (0.7657430730478589, 47, 0.762192964553833, 0.01604294776916504), (0.7657430730478589, 48, 0.7615101337432861, 0.01677703857421875), (0.7783375314861462, 49, 0.7563211917877197, 0.016621828079223633), (0.7506297229219143, 50, 0.7415270805358887, 0.01628899574279785), (0.7506297229219143, 51, 0.7394850254058838, 0.01649618148803711), (0.7506297229219143, 52, 0.7309458255767822, 0.016119956970214844), (0.7506297229219143, 53, 0.719886064529419, 0.01617264747619629), (0.7506297229219143, 54, 0.7128798961639404, 0.016089916229248047), (0.7506297229219143, 55, 0.7044670581817627, 0.016284942626953125), (0.7506297229219143, 56, 0.7035841941833496, 0.015852928161621094), (0.7506297229219143, 57, 0.6993012428283691, 0.015820026397705078), (0.7506297229219143, 58, 0.6970829963684082, 0.01590704917907715), (0.7506297229219143, 59, 0.6941728591918945, 0.015964269638061523), (0.7506297229219143, 60, 0.6927449703216553, 0.016110897064208984), (0.7506297229219143, 61, 0.6852920055389404, 0.016354084014892578), (0.7506297229219143, 62, 0.6810791492462158, 0.016287803649902344), (0.7506297229219143, 63, 0.6814508438110352, 0.015588998794555664), (0.7506297229219143, 64, 0.6764841079711914, 0.016125917434692383), (0.7506297229219143, 65, 0.6757969856262207, 0.016066789627075195), (0.7506297229219143, 66, 0.6777541637420654, 0.016727924346923828), (0.7506297229219143, 67, 0.6745188236236572, 0.015572071075439453), (0.7506297229219143, 68, 0.6692941188812256, 0.0156862735748291), (0.7506297229219143, 69, 0.6686911582946777, 0.016205787658691406), (0.7506297229219143, 70, 0.6567449569702148, 0.01593017578125), (0.7506297229219143, 71, 0.6565186977386475, 0.01587200164794922), (0.7506297229219143, 72, 0.6446199417114258, 0.015542030334472656), (0.7506297229219143, 73, 0.6412620544433594, 0.01621103286743164), (0.7506297229219143, 74, 0.6403100490570068, 0.01629805564880371), (0.7506297229219143, 75, 0.6399698257446289, 0.015466928482055664), (0.7506297229219143, 76, 0.6364998817443848, 0.015560150146484375), (0.7506297229219143, 77, 0.6332178115844727, 0.015592336654663086), (0.7506297229219143, 78, 0.6360898017883301, 0.016124963760375977), (0.7380352644836272, 79, 0.6337490081787109, 0.01565718650817871), (0.7380352644836272, 80, 0.6278948783874512, 0.015771150588989258), (0.7380352644836272, 81, 0.6333520412445068, 0.015962839126586914), (0.7380352644836272, 82, 0.6162681579589844, 0.016041040420532227), (0.7405541561712846, 83, 0.5976419448852539, 0.01597118377685547), (0.7405541561712846, 84, 0.5959019660949707, 0.015456914901733398), (0.7380352644836272, 85, 0.593864917755127, 0.01581120491027832), (0.7380352644836272, 86, 0.5939168930053711, 0.016132116317749023), (0.7380352644836272, 87, 0.5926358699798584, 0.01584482192993164), (0.7380352644836272, 88, 0.5911200046539307, 0.015700101852416992), (0.7380352644836272, 89, 0.5901670455932617, 0.015374898910522461), (0.7380352644836272, 90, 0.5975191593170166, 0.015936851501464844), (0.7380352644836272, 91, 0.5945849418640137, 0.015962839126586914), (0.7405541561712846, 92, 0.583914041519165, 0.01607966423034668), (0.7405541561712846, 93, 0.5843420028686523, 0.01604294776916504), (0.7405541561712846, 94, 0.5830450057983398, 0.015965938568115234), (0.7405541561712846, 95, 0.5707080364227295, 0.015715837478637695), (0.7405541561712846, 96, 0.5692851543426514, 0.015753984451293945), (0.7405541561712846, 97, 0.569159746170044, 0.015874147415161133), (0.7581863979848866, 98, 0.567889928817749, 0.015861034393310547), (0.7581863979848866, 99, 0.5665569305419922, 0.015362977981567383), (0.7581863979848866, 100, 0.5654840469360352, 0.015908002853393555)]
score = []
min_sample_leaf = []
time_created = []
time_classifire = []
for i in arr:
    score.append(i[0])
    min_sample_leaf.append(i[1])
    time_created.append(i[2])
    time_classifire.append(i[3])
best = max(arr, key=lambda x: x[0])
print(best)

low = min(arr, key=lambda x: x[0])
print(low)
plt.figure(figsize=(15,15))
plt.subplot(3,3, 1)
plt.subplots_adjust( top=0.95,bottom=0.35, hspace=0.4)
plt.plot(min_sample_leaf, score)
plt.scatter(best[1], best[0], c='red')
plt.hlines(best[0], 0, best[1], colors='green')
plt.vlines(best[1],low[0], best[0], colors='green')
plt.legend(title=f'min_sample_leaf = {best[1]}\nbest_score = {best[0]}')
plt.title("min_sample_leaf")
plt.subplot(3,3, 2)
plt.plot(min_sample_leaf, time_created)
plt.legend(title=str(best[2]))
plt.title(f'Decision tree CART 80% train data\n\ntime_created')
plt.subplot(3,3, 3)
plt.plot(min_sample_leaf, time_classifire)
plt.title("time_classifire")
arr = [(0.7583081570996979, 1, 3.5464420318603516, 0.05104422569274902), (0.7744209466263847, 2, 2.370413064956665, 0.04835820198059082), (0.7734138972809668, 3, 1.8314788341522217, 0.04676103591918945), (0.7875125881168177, 4, 1.5748889446258545, 0.047808170318603516), (0.7804632426988922, 5, 1.4409809112548828, 0.04667186737060547), (0.7814702920443102, 6, 1.2690529823303223, 0.04564714431762695), (0.7774420946626385, 7, 1.1869521141052246, 0.0444788932800293), (0.7693856998992951, 8, 1.2833750247955322, 0.044355154037475586), (0.7663645518630413, 9, 1.0170509815216064, 0.04329705238342285), (0.7663645518630413, 10, 0.9565639495849609, 0.04236483573913574), (0.7683786505538771, 11, 0.9007568359375, 0.04261922836303711), (0.7673716012084593, 12, 0.8798458576202393, 0.0432279109954834), (0.7724068479355488, 13, 0.8441379070281982, 0.042426109313964844), (0.7653575025176234, 14, 0.8125419616699219, 0.041589975357055664), (0.7683786505538771, 15, 0.7771339416503906, 0.04197406768798828), (0.7452165156092648, 16, 0.7381341457366943, 0.043286800384521484), (0.7613293051359517, 17, 0.7140631675720215, 0.041423797607421875), (0.7653575025176234, 18, 0.6951203346252441, 0.04540896415710449), (0.7653575025176234, 19, 0.6973998546600342, 0.04149127006530762), (0.7613293051359517, 20, 0.66127610206604, 0.04226493835449219), (0.7613293051359517, 21, 0.7519111633300781, 0.041471004486083984), (0.7613293051359517, 22, 0.8217918872833252, 0.05576300621032715), (0.7613293051359517, 23, 0.6458990573883057, 0.04121088981628418), (0.7613293051359517, 24, 0.6309671401977539, 0.04124808311462402), (0.7552870090634441, 25, 0.6111049652099609, 0.040879249572753906), (0.7552870090634441, 26, 0.8829002380371094, 0.059424638748168945), (0.7552870090634441, 27, 0.5953400135040283, 0.04135489463806152), (0.7764350453172205, 28, 0.5632922649383545, 0.03939199447631836), (0.7764350453172205, 29, 0.540229082107544, 0.041024208068847656), (0.7673716012084593, 30, 0.5309340953826904, 0.0404658317565918), (0.7724068479355488, 31, 0.5311312675476074, 0.040726661682128906), (0.7724068479355488, 32, 0.5196781158447266, 0.039602041244506836), (0.7844914400805639, 33, 0.515312910079956, 0.04027891159057617), (0.7844914400805639, 34, 0.512451171875, 0.03924274444580078), (0.7724068479355488, 35, 0.4977121353149414, 0.03920102119445801), (0.7724068479355488, 36, 0.493574857711792, 0.03921675682067871), (0.7583081570996979, 37, 0.4910151958465576, 0.04087972640991211), (0.7583081570996979, 38, 0.4850451946258545, 0.03964996337890625), (0.7583081570996979, 39, 0.47052907943725586, 0.040380001068115234), (0.7583081570996979, 40, 0.4702310562133789, 0.03901386260986328), (0.7683786505538771, 41, 0.46886396408081055, 0.03949713706970215), (0.7683786505538771, 42, 0.457535982131958, 0.039655208587646484), (0.7683786505538771, 43, 0.45247697830200195, 0.03872203826904297), (0.7512588116817724, 44, 0.44658780097961426, 0.04070711135864258), (0.7512588116817724, 45, 0.4449119567871094, 0.03963494300842285), (0.7512588116817724, 46, 0.4546182155609131, 0.08427286148071289), (0.7512588116817724, 47, 0.514167070388794, 0.04083108901977539), (0.7512588116817724, 48, 0.4343428611755371, 0.038923025131225586), (0.7512588116817724, 49, 0.4266812801361084, 0.03867983818054199), (0.7512588116817724, 50, 0.4229087829589844, 0.03866291046142578), (0.7512588116817724, 51, 0.4213728904724121, 0.039324045181274414), (0.7512588116817724, 52, 0.418870210647583, 0.04038596153259277), (0.7512588116817724, 53, 0.4382619857788086, 0.043537139892578125), (0.7512588116817724, 54, 0.41739487648010254, 0.038787126541137695), (0.7512588116817724, 55, 0.4197089672088623, 0.04071187973022461), (0.7512588116817724, 56, 0.4184150695800781, 0.04123997688293457), (0.7512588116817724, 57, 0.42539215087890625, 0.04239487648010254), (0.7512588116817724, 58, 0.42337608337402344, 0.041467905044555664), (0.7512588116817724, 59, 0.4113738536834717, 0.04014921188354492), (0.75730110775428, 60, 0.39228010177612305, 0.038227081298828125), (0.75730110775428, 61, 0.38748788833618164, 0.0387721061706543), (0.7512588116817724, 62, 0.39158010482788086, 0.039865732192993164), (0.7512588116817724, 63, 0.38246893882751465, 0.040296077728271484), (0.7512588116817724, 64, 0.3893120288848877, 0.04033803939819336), (0.7512588116817724, 65, 0.37461304664611816, 0.03945493698120117), (0.7512588116817724, 66, 0.3816680908203125, 0.039505958557128906), (0.7512588116817724, 67, 0.37273502349853516, 0.04131197929382324), (0.7512588116817724, 68, 0.3674178123474121, 0.04091906547546387), (0.7512588116817724, 69, 0.3638589382171631, 0.03970527648925781), (0.7512588116817724, 70, 0.3574860095977783, 0.037715911865234375), (0.7512588116817724, 71, 0.34842491149902344, 0.03750205039978027), (0.7512588116817724, 72, 0.3453800678253174, 0.039350032806396484), (0.7512588116817724, 73, 0.35622334480285645, 0.039786577224731445), (0.7512588116817724, 74, 0.3492882251739502, 0.03825092315673828), (0.7512588116817724, 75, 0.3470580577850342, 0.04669475555419922), (0.7512588116817724, 76, 0.3584449291229248, 0.0419468879699707), (0.7512588116817724, 77, 0.3545491695404053, 0.039691925048828125), (0.7512588116817724, 78, 0.3429679870605469, 0.03862190246582031), (0.7512588116817724, 79, 0.3368709087371826, 0.03926420211791992), (0.7512588116817724, 80, 0.33566784858703613, 0.037882328033447266), (0.7512588116817724, 81, 0.35260987281799316, 0.040509939193725586), (0.7512588116817724, 82, 0.3391838073730469, 0.037937164306640625), (0.7512588116817724, 83, 0.33959126472473145, 0.040068864822387695), (0.7512588116817724, 84, 0.3383491039276123, 0.04024195671081543), (0.7512588116817724, 85, 0.33193016052246094, 0.038823843002319336), (0.7512588116817724, 86, 0.32738685607910156, 0.0373532772064209), (0.7512588116817724, 87, 0.324857234954834, 0.03754591941833496), (0.7512588116817724, 88, 0.3259131908416748, 0.038565874099731445), (0.7512588116817724, 89, 0.32408905029296875, 0.03847312927246094), (0.7512588116817724, 90, 0.3252449035644531, 0.03886222839355469), (0.7512588116817724, 91, 0.32677507400512695, 0.03865385055541992), (0.7512588116817724, 92, 0.32498908042907715, 0.0384058952331543), (0.7512588116817724, 93, 0.3242301940917969, 0.03919792175292969), (0.7512588116817724, 94, 0.3240230083465576, 0.038346052169799805), (0.7512588116817724, 95, 0.32298898696899414, 0.03828692436218262), (0.7512588116817724, 96, 0.3066098690032959, 0.03767895698547363), (0.7512588116817724, 97, 0.2935166358947754, 0.036803245544433594), (0.7512588116817724, 98, 0.29184508323669434, 0.03678297996520996), (0.7512588116817724, 99, 0.2818007469177246, 0.036890268325805664), (0.7512588116817724, 100, 0.28083205223083496, 0.03651571273803711)]
score = []
min_sample_leaf = []
time_created = []
time_classifire = []
for i in arr:
    score.append(i[0])
    min_sample_leaf.append(i[1])
    time_created.append(i[2])
    time_classifire.append(i[3])
best = max(arr, key=lambda x: x[0])
low = min(arr, key=lambda x: x[0])

plt.subplot(3,3, 4)
plt.plot(min_sample_leaf, score)
plt.scatter(best[1], best[0], c='red')
plt.hlines(best[0], 0, best[1], colors='green')
plt.vlines(best[1],low[0], best[0], colors='green')
plt.legend(title=f'min_sample_leaf = {best[1]}\nbest_score = {best[0]}')
plt.title("min_sample_leaf")
plt.subplot(3,3, 5)
plt.plot(min_sample_leaf, time_created)
plt.legend(title=str(best[2]))
plt.title(f'Decision tree CART 50% train data\n\ntime_created')
plt.subplot(3,3, 6)
plt.plot(min_sample_leaf, time_classifire)
plt.title("time_classifire")
arr = [(0.7319068596601637, 1, 1.8173840045928955, 0.08295106887817383), (0.7734424166142227, 2, 1.1288959980010986, 0.07598304748535156), (0.7444933920704846, 3, 0.8477261066436768, 0.07694005966186523), (0.748898678414097, 4, 0.7620828151702881, 0.0760350227355957), (0.7438640654499685, 5, 0.6662068367004395, 0.07495594024658203), (0.7495280050346129, 6, 0.6203210353851318, 0.07166266441345215), (0.7665198237885462, 7, 0.5475270748138428, 0.07204389572143555), (0.7577092511013216, 8, 0.5116050243377686, 0.07181906700134277), (0.7426054122089364, 9, 0.4641709327697754, 0.06817007064819336), (0.762114537444934, 10, 0.3452780246734619, 0.0658409595489502), (0.7375707992448081, 11, 0.3327639102935791, 0.0646200180053711), (0.7312775330396476, 12, 0.32573580741882324, 0.06475329399108887), (0.762114537444934, 13, 0.3076937198638916, 0.06346321105957031), (0.7551919446192574, 14, 0.28673696517944336, 0.06358098983764648), (0.7551919446192574, 15, 0.28191518783569336, 0.06350398063659668), (0.7551919446192574, 16, 0.2770848274230957, 0.06538224220275879), (0.7551919446192574, 17, 0.23683691024780273, 0.06133294105529785), (0.7551919446192574, 18, 0.23125624656677246, 0.06346988677978516), (0.7551919446192574, 19, 0.22186803817749023, 0.06172990798950195), (0.7551919446192574, 20, 0.2118678092956543, 0.06026792526245117), (0.7551919446192574, 21, 0.21352601051330566, 0.06259608268737793), (0.7551919446192574, 22, 0.21254682540893555, 0.10683226585388184), (0.7551919446192574, 23, 0.25139498710632324, 0.06308698654174805), (0.7551919446192574, 24, 0.3315441608428955, 0.06296396255493164), (0.7551919446192574, 25, 0.20177388191223145, 0.06528306007385254), (0.7551919446192574, 26, 0.18147587776184082, 0.060546875), (0.7551919446192574, 27, 0.18199586868286133, 0.06253218650817871), (0.7551919446192574, 28, 0.18004608154296875, 0.06204485893249512), (0.7558212712397735, 29, 0.16638803482055664, 0.063507080078125), (0.7558212712397735, 30, 0.1678297519683838, 0.06051826477050781), (0.7558212712397735, 31, 0.1646111011505127, 0.060595035552978516), (0.7558212712397735, 32, 0.15420794486999512, 0.05981183052062988), (0.7558212712397735, 33, 0.15388703346252441, 0.05843210220336914), (0.7558212712397735, 34, 0.15106987953186035, 0.059754133224487305), (0.7558212712397735, 35, 0.1500871181488037, 0.06004595756530762), (0.7558212712397735, 36, 0.14937710762023926, 0.059010982513427734), (0.7558212712397735, 37, 0.13581609725952148, 0.05933690071105957), (0.7558212712397735, 38, 0.13113117218017578, 0.05907487869262695), (0.7558212712397735, 39, 0.12412905693054199, 0.058502197265625), (0.7558212712397735, 40, 0.12318706512451172, 0.05955767631530762), (0.7558212712397735, 41, 0.12259602546691895, 0.061131954193115234), (0.7558212712397735, 42, 0.12635421752929688, 0.06123185157775879), (0.7558212712397735, 43, 0.12459325790405273, 0.05867481231689453), (0.7558212712397735, 44, 0.11982011795043945, 0.06023883819580078), (0.7558212712397735, 45, 0.12095999717712402, 0.05953407287597656), (0.7558212712397735, 46, 0.1217508316040039, 0.060328006744384766), (0.7558212712397735, 47, 0.12061691284179688, 0.0603330135345459), (0.7558212712397735, 48, 0.32214999198913574, 0.09530806541442871), (0.7558212712397735, 49, 0.12730884552001953, 0.05975818634033203), (0.7558212712397735, 50, 0.1251661777496338, 0.058328866958618164), (0.7558212712397735, 51, 0.1150059700012207, 0.05871725082397461), (0.7558212712397735, 52, 0.11577320098876953, 0.058084726333618164), (0.7558212712397735, 53, 0.11340999603271484, 0.057791948318481445), (0.7558212712397735, 54, 0.11200499534606934, 0.058950185775756836), (0.7558212712397735, 55, 0.11200594902038574, 0.05874300003051758), (0.7558212712397735, 56, 0.11013197898864746, 0.05917716026306152), (0.7558212712397735, 57, 0.11528396606445312, 0.057847023010253906), (0.7558212712397735, 58, 0.1026298999786377, 0.05712103843688965), (0.7558212712397735, 59, 0.1021723747253418, 0.057297706604003906), (0.7558212712397735, 60, 0.10078215599060059, 0.05717206001281738), (0.7558212712397735, 61, 0.10173487663269043, 0.05783700942993164), (0.7558212712397735, 62, 0.0996847152709961, 0.057244300842285156), (0.7558212712397735, 63, 0.09891128540039062, 0.05768394470214844), (0.7558212712397735, 64, 0.09214973449707031, 0.057250022888183594), (0.7558212712397735, 65, 0.09025812149047852, 0.05575084686279297), (0.7558212712397735, 66, 0.08985400199890137, 0.05711007118225098), (0.7558212712397735, 67, 0.08946704864501953, 0.056802988052368164), (0.7558212712397735, 68, 0.08833479881286621, 0.05681204795837402), (0.7558212712397735, 69, 0.08865594863891602, 0.0557100772857666), (0.7558212712397735, 70, 0.08799481391906738, 0.057514190673828125), (0.7558212712397735, 71, 0.08875584602355957, 0.056040287017822266), (0.7558212712397735, 72, 0.09543395042419434, 0.05765795707702637), (0.7558212712397735, 73, 0.08803510665893555, 0.0573420524597168), (0.7558212712397735, 74, 0.08735203742980957, 0.05765509605407715), (0.7558212712397735, 75, 0.08839130401611328, 0.061627864837646484), (0.6406544996853367, 76, 0.09237504005432129, 0.05855107307434082), (0.6406544996853367, 77, 0.08732390403747559, 0.05804109573364258), (0.6406544996853367, 78, 0.08618426322937012, 0.056958913803100586), (0.6406544996853367, 79, 0.0857701301574707, 0.05637669563293457), (0.6406544996853367, 80, 0.0853872299194336, 0.05683493614196777), (0.6406544996853367, 81, 0.08559107780456543, 0.056043148040771484), (0.6406544996853367, 82, 0.08459973335266113, 0.056733131408691406), (0.6406544996853367, 83, 0.08445978164672852, 0.056179046630859375), (0.6406544996853367, 84, 0.08507180213928223, 0.056912899017333984), (0.6406544996853367, 85, 0.08366727828979492, 0.0564727783203125), (0.6406544996853367, 86, 0.08362722396850586, 0.05674099922180176), (0.7061044682190056, 87, 0.06596994400024414, 0.05465412139892578), (0.7061044682190056, 88, 0.06743288040161133, 0.058342933654785156), (0.7061044682190056, 89, 0.06876325607299805, 0.05763888359069824), (0.7061044682190056, 90, 0.06765222549438477, 0.05636096000671387), (0.7061044682190056, 91, 0.06563687324523926, 0.054810285568237305), (0.7061044682190056, 92, 0.06436681747436523, 0.05557608604431152), (0.7061044682190056, 93, 0.06421518325805664, 0.054100990295410156), (0.7061044682190056, 94, 0.06379199028015137, 0.05628681182861328), (0.7061044682190056, 95, 0.06367778778076172, 0.0547330379486084), (0.7061044682190056, 96, 0.06378483772277832, 0.055587053298950195), (0.7061044682190056, 97, 0.06327176094055176, 0.05505800247192383), (0.7061044682190056, 98, 0.06227421760559082, 0.055140018463134766), (0.7061044682190056, 99, 0.0646820068359375, 0.054593801498413086), (0.7061044682190056, 100, 0.0633542537689209, 0.05563688278198242)]
score = []
min_sample_leaf = []
time_created = []
time_classifire = []
for i in arr:
    score.append(i[0])
    min_sample_leaf.append(i[1])
    time_created.append(i[2])
    time_classifire.append(i[3])
best = max(arr, key=lambda x: x[0])
low = min(arr, key=lambda x: x[0])
plt.subplot(3,3, 7)
plt.plot(min_sample_leaf, score)
plt.scatter(best[1], best[0], c='red')
plt.hlines(best[0], 0, best[1], colors='green')
plt.vlines(best[1],low[0], best[0], colors='green')
plt.legend(title=f'min_sample_leaf = {best[1]}\nbest_score = {best[0]}')
plt.title("min_sample_leaf")
plt.subplot(3,3, 8)
plt.plot(min_sample_leaf, time_created)
plt.legend(title=str(best[2]))
plt.title(f'Decision tree CART 20% train data\n\ntime_created')
plt.subplot(3,3, 9)
plt.plot(min_sample_leaf, time_classifire)
plt.title("time_classifire")
plt.show()
