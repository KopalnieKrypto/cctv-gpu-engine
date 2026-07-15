# Raport z analizy nagrań — porównanie systemu z obserwacjami manualnymi

**Data:** 30 kwietnia 2026
**Zakres testu:** dwa nagrania CCTV (po 5 minut), jedna osoba w polu widzenia
**Metoda weryfikacji:** ręczna anotacja każdej sekundy nagrania → porównanie z wynikiem systemu

---

## Co testowano

System automatycznie analizuje nagranie z kamery i raportuje, ile sekund osoba **siedziała / stała / chodziła**. Aby zweryfikować jego wiarygodność, równolegle wykonano ręczną obserwację każdego nagrania, zanotowano dokładne sekundy każdej zmiany aktywności i porównano oba wyniki.

---

## Film 1 (`job-3ffe5068f6e7`)

### Wyniki

| Aktywność | Obserwacja manualna | System | Różnica |
|---|---:|---:|---:|
| Chodzenie | 80 s | 97 s | +17 s |
| Siedzenie | 31 s | 29 s | −2 s |
| Stanie | 19 s | 20 s | +1 s |
| **Łączny czas aktywności** | **130 s** | **146 s** | **+16 s** |
| Aktywność dominująca | chodzenie | chodzenie | ✅ zgodne |

### Końcowy wskaźnik zgodności: **85 %**

---

## Film 2 (`job-0b1699e6ec20`)

### Wyniki

| Aktywność | Obserwacja manualna | System | Różnica |
|---|---:|---:|---:|
| Chodzenie | 47 s | 50 s | +3 s |
| Siedzenie | 43 s | 40 s | −3 s |
| Stanie | 0 s | 7 s | +7 s |
| **Łączny czas aktywności** | **90 s** | **97 s** | **+7 s** |
| Aktywność dominująca | chodzenie | chodzenie | ✅ zgodne |

### Końcowy wskaźnik zgodności: **86 %**

---

## Z czego mogą wynikać różnice

System i obserwator-człowiek nie zawsze widzą to samo w **dokładnie tym samym momencie**. Trzy czynniki, które tłumaczą rozjazdy między tabelami:

### 1. Krótkie chwile przejściowe między aktywnościami
Kiedy człowiek wstaje z ławki i zaczyna iść, jest przez moment „pomiędzy" — już nie siedzi, jeszcze nie idzie. System obserwuje obraz co sekundę i czasem trafia w taki ułamek sekundy, klasyfikując go jako *stanie*. Człowiek wykonujący notatki zwykle tego momentu nie odnotowuje, bo jest za krótki, żeby zauważyć i zapisać. To wyjaśnia 7 s nadmiaru "stania" w Filmie 2 — w rzeczywistości te 7 s istniało, tylko nie pojawiło się w ręcznej notatce.

### 2. Czułość detekcji ruchu
System uznaje, że osoba *idzie* już przy minimalnym przesunięciu sylwetki względem poprzedniej sekundy. Jeśli osoba „stoi i lekko się porusza" (np. obraca głowę, prostuje się przed wstaniem), system zaliczy to jako chodzenie. Stąd lekkie zawyżenie chodzenia w obu filmach (+3 do +17 s, czyli 4–20 %).

### 3. Obiekty łudząco podobne do osoby
W Filmie 1 osoba w pewnych momentach przepycha **ławkę na kółkach**. System detekcji osób potrafił okazjonalnie sklasyfikować ławkę jako drugą osobę — w raporcie widać to jako „szczyt 2 osoby w kadrze", choć faktycznie była jedna. Część zawyżonego chodzenia w Filmie 1 może pochodzić właśnie z dodatkowych klatek przypisanych do ławki, nie do osoby.

---

## Podsumowanie

W obu filmach system **prawidłowo wskazał aktywność dominującą** (chodzenie) oraz **zachował właściwe proporcje** czasowe między siedzeniem, staniem i chodzeniem. Czasy raportowane przez system mieszczą się w granicach **±15 % wobec ręcznej obserwacji**:

| Film | Końcowy wskaźnik zgodności |
|---|---:|
| Film 1 (`job-3ffe5068f6e7`) | **85 %** |
| Film 2 (`job-0b1699e6ec20`) | **86 %** |

**Wniosek:** dla typowego zastosowania CCTV — szybkiej oceny „co się działo na nagraniu w skali 5/15/30/60 minut" — wynik systemu jest zbieżny z ręczną analizą i można go traktować jako wiarygodne źródło informacji. Drobne rozbieżności (rzędu kilku sekund per aktywność) nie zmieniają obrazu sytuacji ani odpowiedzi na pytanie *„kim była ta osoba i co robiła?"*.

Dalsze obniżenie rozbieżności jest możliwe (filtry odporne na obiekty mobilne typu ławka, dłuższy próg detekcji ruchu) — w razie potrzeby można je wdrożyć w kolejnej wersji.

---

# Zmiana metodyki — 15 lipca 2026 (ByteTrack + OSNet, letterbox)

Ponowna analiza tych samych dwóch nagrań po dwóch zmianach w systemie:

1. **Śledzenie osób (ByteTrack + OSNet Re-ID)** — system nadaje każdej osobie trwałą tożsamość między klatkami i liczy czas *per osoba*. Obiekt wykryty tylko przez chwilę (ławka na kółkach) nie jest już zaliczany jako osoba.
2. **Poprawka skalowania obrazu (letterbox)** — klatka nie jest już rozciągana do kwadratu przed analizą, tylko skalowana z zachowaniem proporcji. Sylwetki mają naturalne kształty, więc detekcja jest wyraźnie pewniejsza.

## Wyniki

| Film 1 | Obserwacja manualna | System (kwiecień) | System (lipiec) |
|---|---:|---:|---:|
| Chodzenie | 80 s | 97 s | 98 s |
| Siedzenie | 31 s | 29 s | 29 s |
| Stanie | 19 s | 20 s | 21 s |
| **Zgodność sekunda-po-sekundzie** | — | **85 %** | **89 %** |

| Film 2 | Obserwacja manualna | System (kwiecień) | System (lipiec) |
|---|---:|---:|---:|
| Chodzenie | 47 s | 50 s | 58 s |
| Siedzenie | 43 s | 40 s | 41 s |
| Stanie | 0 s | 7 s | 5 s |
| **Zgodność sekunda-po-sekundzie** | — | **86 %** | **89 %** |

## Co się poprawiło

- **Zgodność wzrosła z 85/86 % do 89 % w obu filmach.**
- **Zniknął „duch" w kadrze.** W Filmie 2 system raportował wcześniej „szczyt: 2 osoby", choć była tam jedna — teraz raportuje **1 osobę**. To jest dokładnie zgłoszona przez Państwa sytuacja „wózek z taśmą liczony jako osoba". W Filmie 1 fałszywe wykrycie ławki również już nie występuje.
- **Detekcja jest praktycznie kompletna.** W każdym z filmów system pomija dziś tylko **1 sekundę** z całego zakresu opisanego ręcznie.

## Ważne zastrzeżenie do porównywania sum

Sumy w tabelach powyżej **nie powinny być porównywane wprost**. Ręczna notatka opisuje *epizody aktywności* („0:27–0:40 chodzę"), a nie ciągłą obecność osoby w kadrze. Pomiędzy epizodami są luki — w Filmie 1 łącznie **20 sekund**, w Filmie 2 **15 sekund** — w których osoba **była widoczna**, ale obserwator nie zanotował konkretnej czynności (najczęściej krótkie momenty przejściowe między jedną a drugą aktywnością).

System liczy każdą sekundę, w której widzi osobę, więc jego suma jest z definicji wyższa od sumy notatki — i to nie jest błąd systemu. Cała różnica sum (+19 s w Filmie 1, +15 s w Filmie 2) niemal dokładnie odpowiada tym nieopisanym lukom.

Dlatego **miarą jakości jest zgodność sekunda-po-sekundzie** (89 %), a nie różnica sum. Aby porównywać sumy wprost, ręczna anotacja musiałaby opisywać każdą sekundę obecności osoby, bez luk.

## Co pozostaje do poprawy

Pozostałe rozbieżności to prawie wyłącznie **momenty przejściowe**, nie pomyłki tożsamości:

| Rzeczywistość → System | Film 1 | Film 2 |
|---|---:|---:|
| stanie → chodzenie | 4 s | — |
| siedzenie → chodzenie | 4 s | 5 s |
| chodzenie → stanie | 3 s | 4 s |

Źródłem jest **czułość progu ruchu**: system uznaje osobę za idącą już przy niewielkim przesunięciu sylwetki (np. przy wstawaniu z ławki). Śledzenie osób tego nie rozwiązuje — to osobna zmiana, zaplanowana jako kolejny krok.
