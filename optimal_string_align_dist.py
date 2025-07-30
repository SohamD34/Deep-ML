# In this problem, you need to implement a function that calculates the Optimal String Alignment (OSA) distance between two given strings. 
# The OSA distance represents the minimum number of edits required to transform one string into another. Each of these operations costs 1 unit.
# Your task is to find the minimum number of edits needed to convert the first string (s1) into the second string (s2).

# In this problem, you need to implement a function that calculates the Optimal String Alignment (OSA) distance between two given strings. 
# The OSA distance represents the minimum number of edits required to transform one string into another. The allowed edit operations are:
#     Insert a character
#     Delete a character
#     Substitute a character
#     Transpose two adjacent characters

# Each of these operations costs 1 unit.
# Your task is to find the minimum number of edits needed to convert the first string (s1) into the second string (s2).
# For example, the OSA distance between the strings caper and acer is 2: one deletion (removing "p") and one transposition (swapping "a" and "c").

def OSA(source: str, target: str) -> int:

    m, n = len(source), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i  # Deleting all characters from source
    for j in range(n + 1):
        dp[0][j] = j  # Inserting all characters to match target
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if source[i - 1] == target[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # Deletion
                dp[i][j - 1] + 1,      # Insertion
                dp[i - 1][j - 1] + cost # Substitution
            )
            if i > 1 and j > 1 and source[i - 2] == target[j - 1] and source[i - 1] == target[j - 2]:
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1) # Transposition
    
    return dp[m][n]