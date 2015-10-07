#ifndef _SINGLETON_H_
#define _SINGLETON_H_

template <typename T>
class Singleton
{
public:
	Singleton(const Singleton&) = delete;
	Singleton(const Singleton&&) = delete;
	Singleton& operator=(const Singleton&) = delete;
	Singleton& operator=(const Singleton&&) = delete;

	static T& getInstance()
	{
		static T instance; // Guaranteed to be destroyed.
		// Instantiated on first use.
		return instance;
	}

protected:
	Singleton() = default;
	virtual ~Singleton() {}
};

#endif

